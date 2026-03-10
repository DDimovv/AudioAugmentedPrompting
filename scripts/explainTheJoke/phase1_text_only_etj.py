import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3OmniForConditionalGeneration, BitsAndBytesConfig

# ---------------- CONFIG ----------------

DATASET_ID = "theblackcat102/joke_explaination"
DATASET_SPLIT = "train"   

OUT_PATH = "cache/joke_explanations_qwen.jsonl"

MODEL_ID = "Qwen/Qwen3-Omni-30B"
MAX_NEW_TOKENS = 120

# -------------- PROMPT -----------------

def build_messages(joke):
    return [
        {
            "role": "system",
            "content": "You are an expert linguist."
        },
        {
            "role": "user",
            "content": f"""Explain the following joke.

Instructions:
- Do NOT explain your analysis process.
- Focus ONLY on why the joke is humorous.
- Mention wordplay, ambiguity, or implied meaning if present.
- If the joke is not based on wordplay, explain the humor mechanism briefly.

Write a concise paragraph (3–6 sentences).

Joke:
{joke}
"""
        }
    ]

# -------------- MAIN -----------------

def main():

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

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    def generate(joke: str) -> str:
        messages = build_messages(joke)

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

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(ds), total=len(ds), desc="Explaining jokes"):
            joke = item.get("joke", "").strip()
            if not joke:
                continue

            explanation = generate(joke)

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
