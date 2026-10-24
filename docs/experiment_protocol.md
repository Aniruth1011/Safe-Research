# Experiment protocol

Train detectors on seen-task trajectories. Use held-out seen-task validation trajectories for layer, architecture, hyperparameter, threshold, and checkpoint selection. Use unseen tasks only once the configuration is finalized for final evaluation. Record the exact task IDs and split seed in the run configuration. Aggregate metrics at trajectory level unless a model explicitly declares per-timestep output.

SAFE's original rollout split and metric functions remain available through the original Hydra path. The cached-feature path records its split version and uses `seen_val` as the development split by default.
