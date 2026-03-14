import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig

# ================= CONFIG =================

DATASET_PATH = "context_situated_pun.csv"   # local file
DATASET_SPLIT = "train"

OUT_PATH = "cache/pun_explanations_qwen.jsonl"

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 120

# ================= PROMPT =================

def build_messages(text):
    return [
        {
            "role": "system",
            "content": "You are an expert linguist."
        },
        {
            "role": "user",
            "content": f"""Explain whether the following text contains a pun.

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
        }
    ]

# ================= MAIN =================

def main():
    if not torch.cuda.is_available():
        raise SystemExit("NO GPU DETECTED")

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"Loading model: {MODEL_ID}")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTextToWaveform.from_pretrained(
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

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def generate(text: str) -> str:
        messages = build_messages(text)

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=prompt,
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

    count = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(ds), total=len(ds), desc="Explaining texts"):
            raw_text = item.get("user_pun")

            if raw_text is None:
                continue

            text = str(raw_text).strip()

            # Skip placeholders / empty-like entries
            if not text or text in {"{}"}:
                continue

            explanation = generate(text)

            out_obj = {
                "id": idx,
                "Explanation": explanation,
            }

            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            f.flush()

            count += 1
            torch.cuda.empty_cache()

    print("Done.")
    print(f"Generated {count} explanations")
    print(f"Output -> {OUT_PATH}")

# ================= RUN =================

if __name__ == "__main__":
    main()

