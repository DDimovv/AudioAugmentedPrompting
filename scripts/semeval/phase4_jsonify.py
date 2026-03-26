import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)

# --------------- CONFIG ----------------

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 200
BATCH_SIZE = 5

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

# ------------------ MODEL ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# runtime workaround for missing attribute bug
if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window"):
    Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = False

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
).eval()

# --------------- HELPERS ----------------

def parse_json_output(decoded: str):
    start = decoded.find("{")
    end = decoded.rfind("}")

    if start == -1 or end == -1:
        return {"Reason": None, "Choice": None}

    try:
        obj = json.loads(decoded[start:end + 1])
        if obj.get("Choice") in {"The text is a pun", "The text is not a pun"}:
            return {
                "Reason": obj.get("Reason"),
                "Choice": obj.get("Choice"),
            }
    except Exception:
        pass

    return {"Reason": None, "Choice": None}


def generate_json_batch(reason_texts):
    prompts = []

    for reason_text in reason_texts:
        prompt_text = JSONIFY_PROMPT.format(reason=reason_text)

        messages = [
            {
                "role": "system",
                "content": "You are a classification system that outputs ONLY valid JSON."
            },
            {"role": "user", "content": prompt_text},
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    inputs = processor(
        text=prompts,
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

    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    generated_parts = []

    for i, input_len in enumerate(input_lengths):
        generated_parts.append(out[i, int(input_len):])

    decoded_batch = processor.batch_decode(
        generated_parts,
        skip_special_tokens=True,
    )

    return [parse_json_output(decoded.strip()) for decoded in decoded_batch]


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def process_split(in_path, out_all, out_split, reason_key):
    with open(in_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    valid_items = [item for item in items if item.get(reason_key)]

    with open(out_all, "a", encoding="utf-8") as f_all, open(out_split, "w", encoding="utf-8") as f_split:
        for batch in tqdm(chunked(valid_items, BATCH_SIZE), desc=f"JSONifying {in_path}"):
            reasons = [item[reason_key] for item in batch]
            parsed_batch = generate_json_batch(reasons)

            for item, parsed in zip(batch, parsed_batch):
                out_obj = {
                    "id": item["id"],
                    "Text": item["Text"],
                    "Reason": parsed["Reason"],
                    "Choice": parsed["Choice"],
                }

                line = json.dumps(out_obj, ensure_ascii=False) + "\n"
                f_all.write(line)
                f_split.write(line)

# ------------------ RUN -----------------

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