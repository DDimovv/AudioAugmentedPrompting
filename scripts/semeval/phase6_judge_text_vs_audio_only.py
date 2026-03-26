import json
import torch
from tqdm import tqdm
from collections import Counter

from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)

# --------------- CONFIG ---------------

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 80
BATCH_SIZE = 5

TEXT_HET = "cache/phase4_text.heterographic.jsonl"
TEXT_HOM = "cache/phase4_text.homographic.jsonl"

AUDIO_ONLY_HET = "cache/phase4_audio_only.heterographic.jsonl"
AUDIO_ONLY_HOM = "cache/phase4_audio_only.homographic.jsonl"

OUT_HET = "cache/phase6_judge_text_vs_audio_only.heterographic.jsonl"
OUT_HOM = "cache/phase6_judge_text_vs_audio_only.homographic.jsonl"

# ---------------- JUDGE PROMPT -----------------

JUDGE_PROMPT = """You are a strict evaluator of linguistic explanations.

Your task:
Given a text and two explanations, decide which explanation better identifies
whether the text is a pun AND explains the linguistic mechanism correctly.

Rules:
- Do NOT prefer an explanation because it appears first.
- Do NOT reward verbosity.
- Prefer correctness, clarity, and accurate identification of wordplay.
- If both explanations are equally good or equally weak, choose a tie.

Return ONLY valid JSON in exactly this format:
{{"Choice": "Explanation 1 is much better" | "Explanation 2 is much better" | "Explanation 1 and 2 are of similar quality",
 "Reason": "<short justification>"}}

Text:
{text}

Explanation 1 (text-only):
{exp1}

Explanation 2 (audio-only):
{exp2}
"""

# ----------------- MODEL (8-bit quantized) -----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window"):
    Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = False

model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()
# ---------------- HELPERS -----------------

def load_map(path):
    with open(path, encoding="utf-8") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def parse_judge_output(decoded: str):
    start = decoded.find("{")
    end = decoded.rfind("}")

    if start != -1 and end != -1:
        try:
            obj = json.loads(decoded[start:end + 1])
            if obj.get("Choice") in {
                "Explanation 1 is much better",
                "Explanation 2 is much better",
                "Explanation 1 and 2 are of similar quality",
            }:
                return {
                    "Choice": obj.get("Choice"),
                    "Reason": obj.get("Reason"),
                }
        except Exception:
            pass

    return {"Choice": "INVALID", "Reason": "Parse failure"}


def generate_judge_batch(prompts):
    chat_texts = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a judge that outputs ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_texts.append(text)

    inputs = processor(
        text=chat_texts,
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

    return [parse_judge_output(decoded.strip()) for decoded in decoded_batch]


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ---------------- RUN -----------------

def run_judge(text_path, audio_only_path, out_path, label):
    print(f"\n=== Judging {label} (text vs audio-only) ===")

    text_items = load_map(text_path)
    audio_only_items = load_map(audio_only_path)

    ids = sorted(set(text_items) & set(audio_only_items))
    votes = Counter()

    with open(out_path, "w", encoding="utf-8") as f:
        for batch_ids in tqdm(chunked(ids, BATCH_SIZE), total=(len(ids) + BATCH_SIZE - 1) // BATCH_SIZE):
            prompts = []
            batch_pairs = []

            for i in batch_ids:
                t = text_items[i]
                a = audio_only_items[i]

                prompt = JUDGE_PROMPT.format(
                    text=t["Text"],
                    exp1=t["Reason"],
                    exp2=a["Reason"],
                )

                prompts.append(prompt)
                batch_pairs.append((i, t, a))

            judges = generate_judge_batch(prompts)

            for (i, t, a), judge in zip(batch_pairs, judges):
                choice = judge.get("Choice", "INVALID")
                votes[choice] += 1

                out = {
                    "id": i,
                    "type": label,
                    "judge": judge,
                    "text_reason": t["Reason"],
                    "audio_only_reason": a["Reason"],
                }

                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    total = sum(votes.values())
    print(f"\nResults for {label} (n={total})")
    for k, v in votes.items():
        pct = (v / total * 100) if total else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")

    print("Wrote:", out_path)

# ----------------- MAIN -----------------

if __name__ == "__main__":
    run_judge(TEXT_HET, AUDIO_ONLY_HET, OUT_HET, "heterographic")
    run_judge(TEXT_HOM, AUDIO_ONLY_HOM, OUT_HOM, "homographic")