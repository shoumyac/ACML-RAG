# Reproducing the paper

## 1. Headline table in seconds (library)

```bash
pip install -e ".[dev]"
python examples/reproduce_headline.py
pytest
```

## 2. Per-model reports from the shipped generations (paper scripts)

The scripts in `code/` look for their inputs in `code/data/`. Either copy the files you need there or change the `DATA` path at the top of a script.

```bash
pip install -r requirements.txt
# controlled panel
for t in qwen2_5_1_5b_instruct qwen2_5_3b_instruct phi_3_5_mini_instruct \
         mistral_7b_instruct_v0_3 qwen2_5_7b_instruct gemma_2_9b_it; do
  python code/stage1_cache.py $t && python code/stage2_report.py $t
done
# naturalistic (SQuAD, real retrieval)
for t in squad_bm25_qwen2_5_7b_instruct squad_dense_qwen2_5_7b_instruct \
         squad_bm25_mistral_7b_instruct_v0_3 squad_dense_mistral_7b_instruct_v0_3; do
  python code/stage1_cache.py $t && python code/naturalistic_report.py $t
done
python code/nat_noise_report.py
python code/make_figures.py
```

Files ending in `_partial.json` are resumable checkpoints and can be ignored.

## 3. Regenerating the generations from scratch

Large-model generations were produced on Colab Pro+ (A100) with the notebooks in `code/colab/`. Each notebook header explains its inputs. Smaller models can be run locally with `code/run_experiment_v2.py` after `code/build_world_v2.py`.

## Environment

Python 3.9 or newer. The library needs numpy, scipy and scikit-learn only. The experiment scripts additionally need torch, transformers and friends (`pip install -e ".[experiments]"`).
