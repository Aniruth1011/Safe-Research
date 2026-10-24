# Adding experiments

1. Edit only `ACTIVE_EXPERIMENT` in `AGENTS.md`.
2. Ask the coding agent to implement the active experiment.
3. Review generated implementation, manifest, and tests.
4. Run the provided training command on cached features.
5. Inspect the run checkpoint, metadata, and `experiments/index.csv`.

For a new fusion architecture, set `model_type` to a new registered name, specify explicit validated layers, and state the mask/output contract. The agent should add `failure_prob/models/<name>.py`, register it, add a YAML config and tests, and leave the manifest at `VALIDATED` until real training succeeds. Never select layers or checkpoints using unseen-task metrics.
