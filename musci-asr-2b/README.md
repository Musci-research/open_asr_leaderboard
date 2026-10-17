# Musci-ASR-2B

This directory contains the independent Open ASR Leaderboard submission for
[`Musci-research/Musci-ASR-2B`](https://huggingface.co/Musci-research/Musci-ASR-2B).
The evaluator runs in the dedicated
[`Musci-research/open-asr-leaderboard-musci-asr-2b`](https://huggingface.co/spaces/Musci-research/open-asr-leaderboard-musci-asr-2b)
Space and uses the `qwen-asr` Transformers backend, fixed English decoding, and
batched inference.

## Run

Set an HF token with access to the model and the leaderboard result bucket, then
run:

```bash
HF_TOKEN=hf_... bash musci-asr-2b/submit_jobs.sh
```

The script submits ten H200 jobs: `ami_cleaned`,
`earnings22_cleaned_aa_chunked`, `gigaspeech_cleaned`, `librispeech` clean and
other, `spgispeech`, `voxpopuli_cleaned_aa`, `monsoon_en_in`, `urgent2024`, and
`urgent2024_clean`. Use `ONLY_DATASETS="urgent2024 urgent2024_clean"` to run a
subset, or `MODEL_REVISION=<commit>` to pin the model revision.

The seven official leaderboard datasets remain the primary YAML result scope;
Monsoon and URGENT2024 are reported as additional datasets and do not change the
seven-set `mean_wer` value in the model repository.
