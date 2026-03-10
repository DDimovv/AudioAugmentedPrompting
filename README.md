# Audio Augmented Prompting

Investigating whether augmenting LLM prompts with spoken audio (TTS) improves pun detection and joke explanation compared to text-only prompts.

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (required — all scripts use 8-bit quantized Qwen3-Omni-30B)
- **Piper TTS** (for audio generation phases)

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download the Piper TTS voice model (~65 MB, only needed once)
python setup_piper.py

# 3. For Context-Situated Puns experiments, place the dataset in the project root:
#    context_situated_pun.csv  (already included)
```

> The voice model is saved to `piper_models/` which the scripts use by default.
> To override the model path, set `PIPER_MODEL=/your/path/model.onnx`.

## Project Structure

There are **three experiment tracks**, each with scripts (`.py`) and notebooks (`.ipynb`):

### 1. SemEval (`scripts/semeval/`)

Uses the [SemEval-2017 Task 7 pun detection dataset](https://huggingface.co/datasets/frostymelonade/SemEval2017-task7-pun-detection) (auto-downloaded from HuggingFace).

### 2. Context-Situated Puns (`scripts/context_situated_puns/`)

Uses a local CSV file `context_situated_pun.csv` (must be placed in the project root).

### 3. Explain the Joke (`scripts/explainTheJoke/`)

Uses the [joke_explaination dataset](https://huggingface.co/datasets/theblackcat102/joke_explaination) (auto-downloaded from HuggingFace).

## Running Order

Each track follows the same pipeline. **Run phases in order** — each phase depends on the output of the previous one.

| Phase | Script     | Description                                     |
| ----- | ---------- | ----------------------------------------------- |
| 1     | `phase1_*` | Text-only inference (Qwen3-Omni)                |
| 2     | `phase2_*` | Audio-only inference (TTS → Qwen3-Omni)         |
| 3     | `phase3_*` | Text + Audio inference (TTS → Qwen3-Omni)       |
| 4     | `phase4_*` | JSONify raw explanations into structured format |
| 5     | `phase5_*` | LLM-as-judge: text-only vs text+audio           |
| 6     | `phase6_*` | LLM-as-judge: text-only vs audio-only           |

**Important:** All scripts must be run **from the project root directory** (they use relative paths to `cache/` and `piper_models/`).

**Example (SemEval):**

```bash
python scripts/semeval/phase1_text_only.py
python scripts/semeval/phase2_audio_only.py
python scripts/semeval/phase3_text_and_audio.py
python scripts/semeval/phase4_jsonify.py
python scripts/semeval/phase5_judge.py
python scripts/semeval/phase6_judge_text_vs_audio_only.py
```

Evaluation script (SemEval only):

```bash
python scripts/semeval/evaluate_phase3.py
```

## Output

All intermediate outputs are written to the `cache/` directory as JSONL files. Final judge results are also in `cache/`.

## Notes

- Phases 2 and 3 require `piper` TTS on your PATH and the voice model downloaded via `python setup_piper.py`.
- Phases 1, 4, 5, 6 are text-only and do not need Piper.
- The notebooks (`.ipynb`) contain the same code as the scripts, with `!pip install` cells for Colab use.
- Scripts use 8-bit quantization (`bitsandbytes`) to fit Qwen3-Omni-30B on a single GPU.
