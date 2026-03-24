import os
import json
import torch
from tqdm import tqdm
import soundfile as sf
import librosa

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig

# ---------------- CONFIG -------------------
HF_DATASET = "frostymelonade/SemEval2017-task7-pun-detection"
HF_SPLIT = "test"

TYPES = {"heterographic", "homographic"}

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 120

AUDIO_DIR = "cache/tts"
AUDIO_EXT = ".wav"

OUT_BASE = "cache/phase3_text_audio_raw"
OUT_ALL = OUT_BASE + ".jsonl"
OUT_HET = OUT_BASE + ".heterographic.jsonl"
OUT_HOM = OUT_BASE + ".homographic.jsonl"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ---------------- PROMPT ----------------
AUDIO_PROMPT_TEMPLATE = """Explain whether the following text contains a pun.

You are given the written text and its spoken audio.

Instructions:
- Do NOT explain your analysis process.
- Do NOT define what a pun is.
- Focus ONLY on the linguistic mechanism.
- If the text is a pun, clearly state:
  • the word or phrase involved
  • the two meanings or sound-based ambiguity
- If it is not a pun, clearly state that no wordplay or ambiguity is present.

Write a concise paragraph (3–6 sentences).

Text:
{text}
"""


def build_messages(text, audio_path):
    return [
        {"role": "system", "content": "You are an expert linguist."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": AUDIO_PROMPT_TEMPLATE.format(text=text)},
            ],
        }
    ]


# ---------------- HELPERS ----------------
def normalize_id(x):
    return str(x).strip() if x else None


def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path)

    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    return wav.astype("float32")


def verify_wavs(audio_dir):
    bad = []
    for fn in os.listdir(audio_dir):
        if not fn.endswith(".wav"):
            continue
        try:
            info = sf.info(os.path.join(audio_dir, fn))
            if info.frames == 0:
                bad.append(fn)
        except Exception:
            bad.append(fn)

    print("Bad wav files:", len(bad))
    assert len(bad) == 0, "Some WAV files are invalid"


def move_inputs_to_model(inputs, model):
    moved = {}
    model_dtype = getattr(model, "dtype", None)

    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                if model_dtype is not None:
                    moved[k] = v.to(model.device, dtype=model_dtype)
                else:
                    moved[k] = v.to(model.device)
            else:
                moved[k] = v.to(model.device)
        else:
            moved[k] = v

    return moved


# ---------------- LOAD DATASET METADATA ONLY ----------------
print("=== Reusing existing TTS files ===")

ds = load_dataset(HF_DATASET, split=HF_SPLIT)

items = []
for r in ds:
    if r["type"] in TYPES:
        items.append({
            "id": normalize_id(r["id"]),
            "text": r["text"],
            "type": r["type"],
            "label": r["label"],
        })

grouped = {}
for x in items:
    grouped.setdefault(x["type"], []).append(x)

items = []
for t in sorted(grouped.keys()):
    items.extend(grouped[t])

verify_wavs(AUDIO_DIR)


# ---------------- PHASE B — QWEN3-OMNI ----------------
print("=== Phase B: Qwen3-Omni inference ===")

torch.set_grad_enabled(False)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()


def generate_reason(text, uid):
    wav_path = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
    messages = build_messages(text, wav_path)

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audio = load_audio(wav_path)

    inputs = processor(
        text=prompt,
        audio=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    inputs = move_inputs_to_model(inputs, model)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=40,
            do_sample=False,
            return_audio=False,
        )

    generated_tokens = out[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()


os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)

with open(OUT_ALL, "w", encoding="utf-8") as fa, \
     open(OUT_HET, "w", encoding="utf-8") as fh, \
     open(OUT_HOM, "w", encoding="utf-8") as fm:

    for it in tqdm(items, desc="Inference"):
        uid = it["id"]
        wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)

        if not os.path.exists(wav):
            print(f"[MISSING WAV] {uid}")
            continue

        try:
            reason = generate_reason(it["text"], uid)
        except Exception as e:
            print(f"[ERROR] {uid}: {e}")
            continue

        obj = {
            "id": uid,
            "Text": it["text"],
            "RawReason": reason,
            "Label": it["label"],
            "Type": it["type"],
        }

        line = json.dumps(obj, ensure_ascii=False) + "\n"
        fa.write(line)

        if it["type"] == "heterographic":
            fh.write(line)
        elif it["type"] == "homographic":
            fm.write(line)

        fa.flush()
        fh.flush()
        fm.flush()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("=== DONE: Text + Audio experiment complete ===")