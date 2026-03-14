import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig

#----------------- CONFIG -----------------

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 150

# Phase-1 / Phase-2 inputs
TEXT_IN = "cache/pun_explanations_qwen.jsonl"
AUDIO_IN = "cache/pun_explanations_qwen_audio.jsonl"
AUDIO_ONLY_IN = "cache/pun_explanations_qwen_audio_only.jsonl"

# Phase-4 outputs
TEXT_OUT = "cache/csp_phase4_text.jsonl"
AUDIO_OUT = "cache/csp_phase4_audio.jsonl"
AUDIO_ONLY_OUT = "cache/csp_phase4_audio_only.jsonl"

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

# ----------------MODEL -----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
).eval()

# ---------------- HELPERS -----------------

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

    print(decoded)
    start, end = decoded.find("{"), decoded.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        obj = json.loads(decoded[start:end + 1])
        if obj.get("Choice") in {
            "The text is a pun",
            "The text is not a pun",
        }:
            return obj
    except Exception:
        pass

    return None


def process_file(in_path, out_path):
    with open(in_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]

    with open(out_path, "w", encoding="utf-8") as f_out:
        for item in tqdm(items, desc=f"JSONifying {in_path}"):
            raw_expl = item.get("Explanation")
            if not raw_expl:
                continue

            parsed = generate_json(raw_expl)
            if not parsed:
                parsed = {"Reason": None, "Choice": None}

            out_obj = {
                "id": item["id"],
                "Reason": parsed["Reason"],
                "Choice": parsed["Choice"],
            }

            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

# ---------------- RUN -----------------

def main():
    process_file(TEXT_IN, TEXT_OUT)
    process_file(AUDIO_IN, AUDIO_OUT)
    process_file(AUDIO_ONLY_IN, AUDIO_ONLY_OUT)

    print("Done.")
    print("TEXT       ->", TEXT_OUT)
    print("AUDIO      ->", AUDIO_OUT)
    print("AUDIO_ONLY ->", AUDIO_ONLY_OUT)

if __name__ == "__main__":
    main()

