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

OUT_BASE = "cache/phase2_audio_only_raw"
OUT_ALL = OUT_BASE + ".jsonl"
OUT_HET = OUT_BASE + ".heterographic.jsonl"
OUT_HOM = OUT_BASE + ".homographic.jsonl"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ---------------- PROMPT ----------------

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
        {"role": "system", "content": "You are an expert linguist."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": AUDIO_ONLY_PROMPT},
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

def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l).get("id") for l in f}


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


#-------------------PHASE B  QWEN3-OMNI (AUDIO ONLY)-------------------
print("=== Phase B: Qwen3-Omni inference (audio-only) ===")

torch.set_grad_enabled(False)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
).eval()

def generate_reason(uid):
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
done_all = load_done_ids(OUT_ALL)

with open(OUT_ALL, "a", encoding="utf-8") as f_all, \
     open(OUT_HET, "a", encoding="utf-8") as f_het, \
     open(OUT_HOM, "a", encoding="utf-8") as f_hom:

    for it in tqdm(items, desc="Phase 2 (Qwen3-Omni, audio-only)"):
        uid = it["id"]
        if not uid or uid in done_all:
            continue

        wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)
        if not os.path.exists(wav):
            continue

        reason = generate_reason(uid)

        out_obj = {
            "id": uid,
            "Text": it["text"],
            "RawReason": reason,
            "Label": it["label"],
            "Type": it["type"],
        }

        line = json.dumps(out_obj, ensure_ascii=False) + "\n"

        f_all.write(line)
        if it["type"] == "heterographic":
            f_het.write(line)
        elif it["type"] == "homographic":
            f_hom.write(line)

        f_all.flush()
        f_het.flush()
        f_hom.flush()

        torch.cuda.empty_cache()

print("Done.")
print(f"ALL -> {OUT_ALL}")
print(f"HET -> {OUT_HET}")
print(f"HOM -> {OUT_HOM}")

