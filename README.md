# CALM-Grade

Repository template for the CALM-Grade paper.

## Contents
- experiment runners
- model prediction outputs
- analysis code
- generated tables and figures

## Recommended workflow
1. Download the original datasets from their official sources.
2. Prepare them using scripts in `data/`.
3. Add API keys to `.env`.
4. Run the experiment scripts in `experiments/`.
5. Store prediction CSVs in `predictions/`.
6. Run `analysis/analyze.py`.

## Dataset sources

- Mohler: https://huggingface.co/datasets/nkazi/MohlerASAG
- DigiKlausur: https://github.com/DigiKlausur/ASAG-Dataset

Do not redistribute the original datasets unless their licenses explicitly permit it.
