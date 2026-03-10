import os
import json
import subprocess
import torch
import soundfile as sf
import librosa
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3OmniForConditionalGeneration, BitsAndBytesConfig

# --------------- CONFIG ---------------

DATASET_ID = "theblackcat102/joke_explaination"
DATASET_SPLIT = "train"

OUT_PATH = "cache/joke_explanations_qwen_audio.jsonl"

MODEL_ID = "Qwen/Qwen3-Omni-30B"
MAX_NEW_TOKENS = 120

# TTS
PIPER_MODEL = os.environ.get("PIPER_MODEL", "piper_models/en_US-lessac-medium.onnx")
AUDIO_DIR = "cache/joke_tts"
AUDIO_EXT = ".wav"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ------------- PROMPT -------------

AUDIO_PROMPT_TEMPLATE = """Explain the following joke.

You are given the written joke and its spoken audio.

Instructions:
- Do NOT explain your analysis process.
- Focus ONLY on why the joke is humorous.
- Mention wordplay, ambiguity, or implied meaning if present.
- If the joke is not based on wordplay, explain the humor mechanism briefly.

Write a concise paragraph (3–6 sentences).

Joke:
{joke}
"""

def build_messages(joke, audio_path):
    return [
        {
            "role": "system",
            "content": "You are an expert linguist."
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": AUDIO_PROMPT_TEMPLATE.format(joke=joke)},
            ],
        }
    ]

# -------------- HELPERS --------------

def normalize_id(idx):
    return f"joke_{idx}"

def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav

def generate_tts(text, uid):
    out_wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)

    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000:
        return True

    p = subprocess.run(
        [
            "piper",
            "--model", PIPER_MODEL,
            "--output_file", out_wav,
        ],
        input=text + "\n",
        text=True,
        capture_output=True,
    )

    return p.returncode == 0 and os.path.exists(out_wav)

# ---------------- MAIN ----------------

def main():
    if not torch.cuda.is_available():
        raise SystemExit("NO GPU DETECTED")

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"Loading model: {MODEL_ID}")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

    # ---- Phase A: TTS ----
    print("=== Generating TTS ===")
    for idx, item in tqdm(enumerate(ds), total=len(ds), desc="TTS"):
        joke = item.get("joke", "").strip()
        if not joke:
            continue
        uid = normalize_id(idx)
        generate_tts(joke, uid)

    # ---- Phase B: Inference ----
    def generate(joke, uid):
        wav_path = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
        messages = build_messages(joke, wav_path)

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        audio = load_audio(wav_path)

        inputs = processor(
            text=prompt,
            audios=[audio],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=40,
                do_sample=False,
                return_audio=False,
            )

        return processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

    print("=== Explaining jokes (Text + Audio) ===")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(ds), total=len(ds), desc="Inference"):
            joke = item.get("joke", "").strip()
            if not joke:
                continue

            uid = normalize_id(idx)
            wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
            if not os.path.exists(wav):
                continue

            explanation = generate(joke, uid)

            out_obj = {
                "id": idx, 
                "Joke": joke,
                "Explanation": explanation,
                "URL": item.get("url"),
                "GoldExplanation": item.get("explanation"),
            }

            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            f.flush()

            torch.cuda.empty_cache()

    print("Done.")
    print(f"Output -> {OUT_PATH}")

# --------------- RUN ----------------

if __name__ == "__main__":
    main()
