# Contributing

Use branches such as `experiment/EXP_001-gated-fusion`, `feature/intermediate-layer-extraction`, or `fix/checkpoint-tracking`. Every experiment needs a unique manifest under `experiments/manifests/`, a resolved config, and CPU tests. Model code must use the shared registry and contract; modify the common engine only for approved shared capabilities.

Do not commit rollouts, cached features, checkpoints, API keys, credentials, or personal filesystem paths. Store large artifacts in the configured output or cluster filesystem. Use seen-task validation for all selection decisions and reserve unseen tasks for final evaluation.

For a pull request, include the experiment ID, tests run, data/split version, reproducibility command, and whether results are synthetic or real. Full VLA evaluation is not suitable for GitHub Actions. On SLURM, run `python scripts/train.py ...` from an allocated job; cluster paths should be supplied as arguments or environment variables, never hardcoded.
