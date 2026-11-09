"""Fixed seeds and split shared by the whole team. Changing anything here changes every result.

Three different seeds, for three different things:
  ROLLOUT_SEED  - OpenVLA LIBERO rollouts (collection/carc/config.sh SEED). With LIBERO's fixed initial states and
                  greedy decoding this makes the rollouts the same for everyone (one shared cache).
  SPLIT         - which tasks are unseen + which seen episodes are train vs seen_val. Frozen as explicit trajectory
                  IDs in the cache's splits.json; the seed below only generated it once.
  TRAIN_SEEDS   - detector training seeds (init + batch order). Every experiment runs all of them.
tests/test_shared_cache.py checks these values match the collection scripts and the training CLI.
"""

ROLLOUT_SEED = 7                 # SAFE / OpenVLA default for run_libero_eval.py

UNSEEN_TASKS = (0, 3, 5)         # = SAFE split_rollouts after seed_everything(0) (verified by running SAFE's code)
SEEN_TRAIN_RATIO = 0.6           # SAFE OpenvlaDatasetConfig.seen_train_ratio
SPLIT_SEED = 0                   # used once by collection/build_cache.py; afterwards splits.json is the source of truth

TRAIN_SEEDS = (0, 1, 2)          # SAFE runs seeds 0/1/2
