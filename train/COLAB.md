# Train it on a free GPU (about 45 minutes, $0)

1. colab.research.google.com -> New notebook -> Runtime -> Change runtime type -> T4 GPU.
2. Llama weights are gated: accept the license at huggingface.co/meta-llama/Llama-3.2-1B-Instruct, then make a read token
   (huggingface.co/settings/tokens). If you'd rather skip that, use `--model Qwen/Qwen2.5-1.5B-Instruct` (no gate).
3. Upload the project zip (left sidebar -> Files), then in cells:

```
!unzip -q healthcheck-bot.zip -d hb && cd hb
!pip -q install transformers peft accelerate pypdf
from huggingface_hub import login; login()          # paste token (skip for Qwen)
!python scripts/gen_sft_data.py --n 3000
!python scripts/eval_local_model.py --model meta-llama/Llama-3.2-1B-Instruct --out evals/results/local_base.json
!python scripts/train_lora.py --model meta-llama/Llama-3.2-1B-Instruct --limit 64     # smoke test first (2 min)
!python scripts/train_lora.py --model meta-llama/Llama-3.2-1B-Instruct               # real run
!python scripts/eval_local_model.py --model meta-llama/Llama-3.2-1B-Instruct --adapter train/adapter --out evals/results/local_tuned.json
```
4. Download `evals/results/local_base.json` and `local_tuned.json` and paste them to me.

Honest caveats: training data is synthetic and templated; "ood" holds out payers and phrasings but is the same family;
"eval" uses different wording from training but is still synthetic. Photos are not covered (no OCR step).
