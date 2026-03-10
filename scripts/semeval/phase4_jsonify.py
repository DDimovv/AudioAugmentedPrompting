import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3OmniForConditionalGeneration, BitsAndBytesConfig

# --------------- CONFIG ----------------

MODEL_ID = "Qwen/Qwen3-Omni-30B"
MAX_NEW_TOKENS = 200

TEXT_HET_IN = "cache/phase1_text_only_raw.heterographic.jsonl"
TEXT_HOM_IN = "cache/phase1_text_only_raw.homographic.jsonl"

AUDIO_HET_IN = "cache/phase3_text_audio_raw.heterographic.jsonl"
AUDIO_HOM_IN = "cache/phase3_text_audio_raw.homographic.jsonl"

AUDIO_ONLY_HET_IN = "cache/phase2_audio_only_raw.heterographic.jsonl"
AUDIO_ONLY_HOM_IN = "cache/phase2_audio_only_raw.homographic.jsonl"

TEXT_ALL_OUT = "cache/phase4_text.jsonl"
TEXT_HET_OUT = "cache/phase4_text.heterographic.jsonl"
TEXT_HOM_OUT = "cache/phase4_text.homographic.jsonl"

AUDIO_ALL_OUT = "cache/phase4_audio.jsonl"
AUDIO_HET_OUT = "cache/phase4_audio.heterographic.jsonl"
AUDIO_HOM_OUT = "cache/phase4_audio.homographic.jsonl"

AUDIO_ONLY_ALL_OUT = "cache/phase4_audio_only.jsonl"
AUDIO_ONLY_HET_OUT = "cache/phase4_audio_only.heterographic.jsonl"
AUDIO_ONLY_HOM_OUT = "cache/phase4_audio_only.homographic.jsonl"

JSONIFY_PROMPT = """You are a classification system.

STRICT RULES:
- Output ONLY a single JSON object
- No markdown
- No extra text
- Use EXACT strings for Choice

Task:
You are given an explanation about whether a text is a pun.

1. Decide if the explanation concludes the text IS or IS NOT a pun
2. Rewrite the explanation into a clean, concise reason

Output format (EXACT):
{{
  "Reason": "<clean explanation>",
  "Choice": "The text is a pun" | "The text is not a pun"
}}

Explanation:
{reason}
"""

# ------------------MODEL----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
).eval()

#  ---------------HELPERS---------------

def generate_json(reason_text: str):
    prompt_text = JSONIFY_PROMPT.format(reason=reason_text)

    messages = [
        {"role": "system", "content": "You are a classification system that outputs ONLY valid JSON."},
        {"role": "user", "content": prompt_text},
    ]

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
            do_sample=False,
            return_audio=False,
        )

    decoded = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    start, end = decoded.find("{"), decoded.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        obj = json.loads(decoded[start:end+1])
        if obj.get("Choice") in {"The text is a pun", "The text is not a pun"}:
            return obj
    except Exception:
        pass

    return None


def process_split(in_path, out_all, out_split, reason_key):
    with open(in_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]

    with open(out_all, "a", encoding="utf-8") as f_all, \
         open(out_split, "w", encoding="utf-8") as f_split:

        for item in tqdm(items, desc=f"JSONifying {in_path}"):
            raw_reason = item.get(reason_key)
            if not raw_reason:
                continue

            parsed = generate_json(raw_reason)
            if not parsed:
                parsed = {"Reason": None, "Choice": None}

            out_obj = {
                "id": item["id"],
                "Text": item["Text"],
                "Reason": parsed["Reason"],
                "Choice": parsed["Choice"],
            }

            line = json.dumps(out_obj, ensure_ascii=False) + "\n"
            f_all.write(line)
            f_split.write(line)

# ------------------RUN -----------------
def main():
    # TEXT
    process_split(TEXT_HET_IN, TEXT_ALL_OUT, TEXT_HET_OUT, "RawReason")
    process_split(TEXT_HOM_IN, TEXT_ALL_OUT, TEXT_HOM_OUT, "RawReason")

    # AUDIO (text + audio)
    process_split(AUDIO_HET_IN, AUDIO_ALL_OUT, AUDIO_HET_OUT, "RawReason")
    process_split(AUDIO_HOM_IN, AUDIO_ALL_OUT, AUDIO_HOM_OUT, "RawReason")

    # AUDIO ONLY
    process_split(AUDIO_ONLY_HET_IN, AUDIO_ONLY_ALL_OUT, AUDIO_ONLY_HET_OUT, "RawReason")
    process_split(AUDIO_ONLY_HOM_IN, AUDIO_ONLY_ALL_OUT, AUDIO_ONLY_HOM_OUT, "RawReason")

    print("Done.")
    print("TEXT       ->", TEXT_ALL_OUT)
    print("AUDIO      ->", AUDIO_ALL_OUT)
    print("AUDIO_ONLY ->", AUDIO_ONLY_ALL_OUT)

if __name__ == "__main__":
    main()
