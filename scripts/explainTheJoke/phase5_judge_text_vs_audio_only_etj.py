import json
import torch
from tqdm import tqdm
from collections import Counter

from transformers import AutoProcessor, AutoModelForTextToWaveform, BitsAndBytesConfig

# ---------------- CONFIG ----------------

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_NEW_TOKENS = 120

TEXT_PATH  = "cache/joke_explanations_qwen.jsonl"
AUDIO_ONLY_PATH = "cache/joke_explanations_qwen_audio_only.jsonl"

OUT_PATH = "cache/joke_judge_text_vs_audio_only.jsonl"

# ---------------- SCORING PROMPT ----------------

JUDGE_PROMPT = """You are a strict evaluator of joke explanations.

Task:
Evaluate how well the explanation explains WHY the joke is humorous.

Evaluation criteria:
- Correct identification of the humor mechanism (e.g., wordplay, ambiguity, violated expectation)
- Clarity and correctness
- No hallucinated or irrelevant mechanisms

Score the explanation on a scale from 1 to 5:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Return ONLY valid JSON in exactly this format:
{{"Score": 1 | 2 | 3 | 4 | 5,
 "Reason": "<short justification>"}}

Joke:
{joke}

Explanation:
{exp}
"""

# --------------- MODEL ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForTextToWaveform.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
).eval()

# --------------- HELPERS ----------------

def load_map(path):
    with open(path, encoding="utf-8") as f:
        return {obj["id"]: obj for obj in map(json.loads, f)}

def safe_json_extract(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None

def generate_score(prompt: str):
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

    parsed = safe_json_extract(decoded)
    if parsed is None:
        return {"Score": 0, "Reason": "Parse failure"}

    return parsed

# --------------- RUN -----------------

def run_judge():
    print("=== Loading explanations ===")

    text_items = load_map(TEXT_PATH)
    audio_only_items = load_map(AUDIO_ONLY_PATH)

    ids = sorted(set(text_items) & set(audio_only_items))
    votes = Counter()

    print(f"Scoring {len(ids)} joke pairs (text vs audio-only)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in tqdm(ids, desc="Judging"):
            t = text_items[i]
            a = audio_only_items[i]

            judge_text = generate_score(
                JUDGE_PROMPT.format(
                    joke=t["Joke"],
                    exp=t["Explanation"],
                )
            )

            judge_audio_only = generate_score(
                JUDGE_PROMPT.format(
                    joke=a["Joke"],
                    exp=a["Explanation"],
                )
            )

            s_text = int(judge_text.get("Score", 0))
            s_audio_only = int(judge_audio_only.get("Score", 0))

            if s_text > s_audio_only:
                final = "text"
            elif s_audio_only > s_text:
                final = "audio_only"
            else:
                final = "tie"

            votes[final] += 1

            out = {
                "id": i,
                "text_score": s_text,
                "audio_only_score": s_audio_only,
                "final_decision": final,
                "text_judge": judge_text,
                "audio_only_judge": judge_audio_only,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ------------- STATS -------------

    total = sum(votes.values())
    print("\n=== RESULTS ===")
    for k, v in votes.items():
        pct = (v / total * 100) if total else 0.0
        print(f"{k}: {v} ({pct:.1f}%)")

    print("\nWrote:", OUT_PATH)

# ------------- MAIN --------------

if __name__ == "__main__":
    run_judge()

