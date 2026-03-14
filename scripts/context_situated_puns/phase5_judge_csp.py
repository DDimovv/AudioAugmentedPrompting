import json
import torch
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig

# ================= CONFIG =================

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 80

DATASET_PATH = "context_situated_pun.csv"
DATASET_SPLIT = "train"

TEXT_JSONL = "cache/csp_phase4_text.jsonl"
AUDIO_JSONL = "cache/csp_phase4_audio.jsonl"

OUT_PATH = "cache/csp_phase5_judge_text_vs_audio.jsonl"

TEXT_FIELD = "user_pun"

# ================= JUDGE PROMPT =================

JUDGE_PROMPT = """You are a strict evaluator of linguistic interpretations.

Your task:
Given a text and two structured interpretations, decide which interpretation
better determines whether the text is a pun and explains the linguistic mechanism.
Decide if Explanation 1 is better than Explanation 2, Explanation 2 is better than explanation 1 and if both explanations are of similar quality

Rules:
- Ignore writing style and verbosity.
- Judge only correctness of the decision and mechanism.
- If both interpretations are equally correct or equally incorrect, choose a tie.

Return ONLY valid JSON in exactly this format:
{{"Choice": "Explanation 1 is much better" | "Explanation 2 is much better" | "Explanation 1 and 2 are of similar quality",
 "Reason": "<short justification>"}}

Text:
{text}

Explanation 1:
Decision: {choice1}
Reason: {reason1}

Explanation 2:
Decision: {choice2}
Reason: {reason2}
"""

# ================= MODEL =================

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
).eval()

# ================= HELPERS =================

def load_json_map(path):
    with open(path, encoding="utf-8") as f:
        return {x["id"]: x for x in map(json.loads, f)}

def load_text_map():
    ds = load_dataset(
        "csv" if DATASET_PATH.endswith(".csv") else "json",
        data_files=DATASET_PATH,
        split=DATASET_SPLIT,
    )

    text_map = {}
    for idx, item in enumerate(ds):
        raw = item.get(TEXT_FIELD)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text or text in {"{}", "{ }", "null", "None"}:
            continue
        text_map[idx] = text

    return text_map

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

    start, end = decoded.find("{"), decoded.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(decoded[start:end + 1])
        except Exception:
            pass

    return {"Choice": "INVALID", "Reason": "Parse failure"}

# ================= RUN =================

def main():
    print("Loading original texts\u2026")
    text_map = load_text_map()

    print("Loading normalized explanations\u2026")
    text_exp = load_json_map(TEXT_JSONL)
    audio_exp = load_json_map(AUDIO_JSONL)

    ids = sorted(set(text_map) & set(text_exp) & set(audio_exp))
    votes = Counter()

    print(f"Judging {len(ids)} aligned items")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in tqdm(ids):
            prompt = JUDGE_PROMPT.format(
                text=text_map[i],
                choice1=text_exp[i]["Choice"],
                reason1=text_exp[i]["Reason"],
                choice2=audio_exp[i]["Choice"],
                reason2=audio_exp[i]["Reason"],
            )

            judge = generate_judge(prompt)
            votes[judge.get("Choice", "INVALID")] += 1

            out = {
                "id": i,
                "judge": judge,
                "text_interpretation": text_exp[i],
                "audio_interpretation": audio_exp[i],
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    total = sum(votes.values())
    print("\n=== RESULTS ===")
    for k, v in votes.items():
        print(f"{k}: {v} ({v/total*100:.1f}%)")

    print("Wrote:", OUT_PATH)

# ================= MAIN =================

if __name__ == "__main__":
    main()

