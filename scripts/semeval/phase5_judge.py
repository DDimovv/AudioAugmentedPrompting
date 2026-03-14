import json
import torch
from tqdm import tqdm
from collections import Counter

from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig

#--------------- CONFIG ---------------

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 80

TEXT_HET = "cache/phase4_text.heterographic.jsonl"
TEXT_HOM = "cache/phase4_text.homographic.jsonl"

AUDIO_HET = "cache/phase4_audio.heterographic.jsonl"
AUDIO_HOM = "cache/phase4_audio.homographic.jsonl"

OUT_HET = "cache/phase5_judge_text_vs_audio.heterographic.jsonl"
OUT_HOM = "cache/phase5_judge_text_vs_audio.homographic.jsonl"

# ----------------JUDGE PROMPT -----------------

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

Explanation 1:
{exp1}

Explanation 2:
{exp2}
"""

# ----------------- MODEL (8-bit quantized) -----------------

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

def load_map(path):
    with open(path, encoding="utf-8") as f:
        return {x["id"]: x for x in map(json.loads, f)}

def generate_judge(prompt: str):
    messages = [
        {"role": "system", "content": "You are a judge that outputs ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
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
    # robust JSON extraction
    start, end = decoded.find("{"), decoded.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(decoded[start:end + 1])
        except Exception:
            pass

    return {"Choice": "INVALID", "Reason": "Parse failure"}

# ---------------- RUN -----------------

def run_judge(text_path, audio_path, out_path, label):
    print(f"\n=== Judging {label} ===")

    text_items = load_map(text_path)
    audio_items = load_map(audio_path)

    ids = sorted(set(text_items) & set(audio_items))
    votes = Counter()

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(ids):
            t = text_items[i]
            a = audio_items[i]

            prompt = JUDGE_PROMPT.format(
                text=t["Text"],
                exp1=t["Reason"],
                exp2=a["Reason"],
            )

            judge = generate_judge(prompt)
            choice = judge.get("Choice", "INVALID")
            votes[choice] += 1

            out = {
                "id": i,
                "type": label,
                "judge": judge,
                "text_reason": t["Reason"],
                "audio_reason": a["Reason"],
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ------- PRINT STATS ---------
    total = sum(votes.values())
    print(f"\nResults for {label} (n={total})")
    for k, v in votes.items():
        pct = (v / total * 100) if total else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")

    print("Wrote:", out_path)

# ----------------- MAIN -----------------

if __name__ == "__main__":
    run_judge(TEXT_HET, AUDIO_HET, OUT_HET, "heterographic")
    run_judge(TEXT_HOM, AUDIO_HOM, OUT_HOM, "homographic")

