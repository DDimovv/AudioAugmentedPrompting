"""Download the Piper TTS voice model required for audio experiments."""

import os
import urllib.request

MODEL_DIR = "piper_models"
BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices"
    "/resolve/v1.0.0/en/en_US/lessac/medium"
)
FILES = [
    "en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json",
]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for fname in FILES:
        dest = os.path.join(MODEL_DIR, fname)
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Saved to {dest} ({size_mb:.1f} MB)")

    print("Piper voice model ready.")


if __name__ == "__main__":
    main()
