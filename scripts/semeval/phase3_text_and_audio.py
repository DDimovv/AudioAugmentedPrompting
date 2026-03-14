import os
import json
import subprocess
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

PIPER_MODEL = os.environ.get("PIPER_MODEL", "piper_models/en_US-lessac-medium.onnx")
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
  \u2022 the word or phrase involved
  \u2022 the two meanings or sound-based ambiguity
- If it is not a pun, clearly state that no wordplay or ambiguity is present.

Write a concise paragraph (3\u20136 sentences).

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
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav



#------------------PHASE A - OFFLINE TTS----------------
print("=== Phase A: Generating TTS ===")

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

    if p.returncode != 0:
        print(f"[PIPER ERROR] {uid}")
        print(p.stderr)
        return False

    if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 1000:
        return False

    return True

ok = 0
for it in tqdm(items, desc="TTS"):
    if generate_tts(it["text"], it["id"]):
        ok += 1

print(f"TTS generated for {ok}/{len(items)} items")

# ---------------VERIFY WAVS---------------
bad = []
for fn in os.listdir(AUDIO_DIR):
    try:
        info = sf.info(os.path.join(AUDIO_DIR, fn))
        if info.frames == 0:
            bad.append(fn)
    except:
        bad.append(fn)

print("Bad wav files:", len(bad))
assert len(bad) == 0, "Some WAV files are invalid"


#-------------------PHASE B \u2014 QWEN3-OMNI-------------------
print("=== Phase B: Qwen3-Omni inference ===")

device = "cuda"
torch.set_grad_enabled(False)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
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
        audios=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

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

os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)

with open(OUT_ALL, "w", encoding="utf-8") as fa, \
     open(OUT_HET, "w", encoding="utf-8") as fh, \
     open(OUT_HOM, "w", encoding="utf-8") as fm:

    for it in tqdm(items, desc="Inference"):
        uid = it["id"]
        wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
        if not os.path.exists(wav):
            continue

        reason = generate_reason(it["text"], uid)

        obj = {
            "id": uid,
            "Text": it["text"],
            "RawReason": reason,
            "Label": it["label"],
            "Type": it["type"],
        }

        line = json.dumps(obj, ensure_ascii=False) + "\n"
        fa.write(line)
        (fh if it["type"] == "heterographic" else fm).write(line)

print("=== DONE: Text + Audio experiment complete ===")

