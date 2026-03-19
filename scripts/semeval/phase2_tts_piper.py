import os
import wave
import traceback
from tqdm import tqdm
from datasets import load_dataset
from piper import PiperVoice

HF_DATASET = "frostymelonade/SemEval2017-task7-pun-detection"
HF_SPLIT = "test"
TYPES = {"heterographic", "homographic"}

PIPER_MODEL = os.environ.get("PIPER_MODEL", "piper_models/en_US-lessac-medium.onnx")
AUDIO_DIR = "cache/tts"
AUDIO_EXT = ".wav"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("cache", exist_ok=True)

def normalize_id(x):
    return str(x).strip() if x else None

def generate_tts(voice, text, uid):
    out_wav = os.path.join(AUDIO_DIR, uid + AUDIO_EXT)

    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000:
        return True

    try:
        with wave.open(out_wav, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
    except Exception:
        print(f"[PIPER ERROR] {uid}")
        traceback.print_exc()
        return False

    return os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000

def main():
    print("=== Phase A: Generating TTS with Piper Python API ===")

    voice = PiperVoice.load(PIPER_MODEL)
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

    ok = 0
    for it in tqdm(items, desc="TTS"):
        if generate_tts(voice, it["text"], it["id"]):
            ok += 1

    print(f"TTS generated for {ok}/{len(items)} items")

if __name__ == "__main__":
    main()