import os
import json
import subprocess
import torch
import soundfile as sf
import librosa
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3OmniForConditionalGeneration, BitsAndBytesConfig

# ================= CONFIG =================

DATASET_PATH = "context_situated_pun.csv"   # local file
DATASET_SPLIT = "train"

OUT_PATH = "cache/pun_explanations_qwen_audio_only.jsonl"

MODEL_ID = "Qwen/Qwen3-Omni-30B"
MAX_NEW_TOKENS = 120

# TTS
PIPER_MODEL = os.environ.get("PIPER_MODEL", "piper_models/en_US-lessac-medium.onnx")
AUDIO_DIR = "cache/pun_tts"
AUDIO_EXT = ".wav"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ================= PROMPT =================

AUDIO_ONLY_PROMPT = """Listen to the following spoken audio and explain whether it contains a pun.

Instructions:
- Do NOT explain your analysis process.
- Do NOT define what a pun is.
- Focus ONLY on the linguistic mechanism.
- If the audio contains a pun, clearly state:
  \u2022 the word or phrase involved
  \u2022 the two meanings or sound-based ambiguity
- If it is not a pun, clearly state that no wordplay or ambiguity is present.

Write a concise paragraph (3\u20136 sentences).
"""

def build_messages(audio_path):
    return [
        {
            "role": "system",
            "content": "You are an expert linguist."
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": AUDIO_ONLY_PROMPT},
            ],
        }
    ]

# ================= HELPERS =================

def normalize_id(idx):
    return f"pun_{idx}"

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

def valid_user_pun(item):
    raw = item.get("user_pun")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text in {"{}", "{ }", "null", "None"}:
        return None
    return text

# ================= MAIN =================

def main():

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

    print(f"Loading local dataset: {DATASET_PATH}")
    ds = load_dataset(
        "csv" if DATASET_PATH.endswith(".csv") else "json",
        data_files=DATASET_PATH,
        split=DATASET_SPLIT,
    )

    # ---- Phase A: TTS ----
    print("=== Generating TTS ===")
    tts_count = 0
    for idx, item in tqdm(enumerate(ds), total=len(ds), desc="TTS"):
        text = valid_user_pun(item)
        if not text:
            continue

        uid = normalize_id(idx)
        if generate_tts(text, uid):
            tts_count += 1

    # ---- Phase B: Inference (audio only) ----
    def generate(uid):
        wav_path = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
        messages = build_messages(wav_path)

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

    print("=== Explaining texts (Audio Only) ===")
    count = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(ds), total=len(ds), desc="Inference"):
            text = valid_user_pun(item)
            if not text:
                continue

            uid = normalize_id(idx)
            wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
            if not os.path.exists(wav):
                continue

            explanation = generate(uid)

            out_obj = {
                "id": idx,
                "Explanation": explanation,
            }

            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            f.flush()

            count += 1
            torch.cuda.empty_cache()

    print("Done.")
    print(f"Generated {count} audio-only explanations")
    print(f"Output \u2192 {OUT_PATH}")

# ---------------- RUN -----------------

if __name__ == "__main__":
    main()
