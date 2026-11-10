"""T6: layer-index and hook verification for OpenVLA-7B (libero_10) -- GPU only.

Run on a GPU node inside the *collection* env (vla-safe/openvla fork, torch 2.2.0,
transformers 4.40.1), never on CPU / login nodes:

    export OPENVLA_FORK_DIR=/path/to/vla-safe/openvla
    export VLA_CKPT=openvla/openvla-7b-finetuned-libero-10     # or a local path
    export MUJOCO_GL=egl
    python -m pytest tests/gpu/test_layer_mapping.py -s -q

Module paths (verified in the fork's source, not guessed):
  model                      = OpenVLAForActionPrediction (prismatic/extern/hf/modeling_prismatic.py)
  model.language_model       = LlamaForCausalLM (AutoModelForCausalLM.from_config)
  ...language_model.model    = LlamaModel
  ...language_model.model.layers[i]  = LlamaDecoderLayer, i = 0..31  (block k = layers[k-1])
  ...language_model.model.norm       = LlamaRMSNorm (final norm)
Policy path mirrors experiments/robot/openvla_utils.get_vla_action:
  preprocess_image(center_crop=True) -> processor(prompt, image) -> predict_action(
  do_sample=False, return_dict_in_generate=True, output_hidden_states=True).

Checks (one observation, 7 generation states j = 0..6):
  1. hidden_states[j][k][0,-1] == forward-hook output of layers[k-1][0,-1], k = 1..31 (exact)
  2. final-norm forward-hook output == hidden_states[j][32][0,-1] (exact), and
     RMSNorm(raw block 32, captured by pre-hook on norm) vs hidden_states[j][32][0,-1]:
     max abs / max rel error are REPORTED; asserted against VLA_NORM_ATOL (default 0.0 = exact).
     Loosen only after reading the reported numbers.
  3. the 7 greedy action-token ids are identical with and without hooks.
Report is written to $VLA_T6_REPORT (default ./t6_report.json).
Image source: LIBERO libero_10 task 0, init state 0, 10 dummy steps (as num_steps_wait).
Task label is task.language; unnorm key follows run_libero_eval.py:121-132.
attention implementation defaults to flash_attention_2 (VLA_ATTN overrides) and is recorded in the report.
Set VLA_T6_SYNTH_IMAGE=1 to use a fixed random image instead (layer mapping does not depend on it).
"""
import json
import os
import sys
import types

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU"),
    pytest.mark.skipif(not os.environ.get("VLA_CKPT"), reason="set VLA_CKPT to the OpenVLA checkpoint"),
    pytest.mark.skipif(not os.environ.get("OPENVLA_FORK_DIR"), reason="set OPENVLA_FORK_DIR to the vla-safe/openvla clone"),
]

NUM_BLOCKS = 32
ACTION_DIM = 7
TASK_SUITE_NAME = "libero_10"
TASK_ID = 0
ATTN_IMPL = os.environ.get("VLA_ATTN", "flash_attention_2")  # fork/SAFE default (run_libero_eval.py:72)


@pytest.fixture(scope="module")
def fork():
    sys.path.insert(0, os.environ["OPENVLA_FORK_DIR"])
    from experiments.robot import openvla_utils as ou
    return ou


@pytest.fixture(scope="module")
def model_and_processor(fork):
    cfg = types.SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=os.environ["VLA_CKPT"],
        attn_implementation=ATTN_IMPL,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
    )
    model = fork.get_vla(cfg).eval()
    processor = fork.get_processor(cfg)
    # Copied from the fork's run_libero_eval.py:121-132 (cfg.unnorm_key = cfg.task_suite_name, then _no_noops fallback)
    unnorm_key = TASK_SUITE_NAME
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`! {list(model.norm_stats)}"
    return model, processor, unnorm_key


@pytest.fixture(scope="module")
def observation():
    """(image, task_label): image at the get_libero_image output boundary (224x224x3 uint8, pre center-crop);
    task_label = task.language, as in the fork (get_libero_env returns task.language)."""
    import numpy as np
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
    suite = benchmark.get_benchmark_dict()[TASK_SUITE_NAME]()
    task = suite.get_task(TASK_ID)
    task_label = task.language
    if os.environ.get("VLA_T6_SYNTH_IMAGE") == "1":
        return np.random.RandomState(0).randint(0, 255, (224, 224, 3), dtype=np.uint8), task_label
    init_states = suite.get_task_init_states(TASK_ID)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    assert task_description == task_label
    try:
        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(10):
            obs, *_ = env.step(get_libero_dummy_action("openvla"))
        return get_libero_image(obs, 224), task_label
    finally:
        env.close()


def _locate(model):
    lm = model.language_model
    llama = lm.model
    layers = llama.layers
    norm = llama.norm
    assert type(llama).__name__ == "LlamaModel", type(llama)
    assert len(layers) == NUM_BLOCKS, len(layers)
    assert type(layers[0]).__name__ == "LlamaDecoderLayer"
    assert type(norm).__name__ == "LlamaRMSNorm", type(norm)
    return layers, norm


def _predict(fork, model, processor, unnorm_key, image_uint8, task_label, output_hidden_states=True):
    image = fork.preprocess_image(image_uint8, True)
    prompt = fork.get_vla_prompt(task_label, os.environ["VLA_CKPT"])
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        _, out = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=output_hidden_states,
            output_logits=True,
        )
    return out


def test_layer_mapping(fork, model_and_processor, observation):
    model, processor, unnorm_key = model_and_processor
    layers, norm = _locate(model)
    image_uint8, task_label = observation

    # --- reference: no hooks ---------------------------------------------------------------
    ref = _predict(fork, model, processor, unnorm_key, image_uint8, task_label)
    ref_tokens = ref["sequences"][0, -ACTION_DIM:].cpu().tolist()
    ref_nohs = _predict(fork, model, processor, unnorm_key, image_uint8, task_label, output_hidden_states=False)
    ref_nohs_tokens = ref_nohs["sequences"][0, -ACTION_DIM:].cpu().tolist()

    # --- hooked run ------------------------------------------------------------------------
    block_out = {k: [] for k in range(1, NUM_BLOCKS + 1)}   # k (1-based) -> list over steps of [4096]
    norm_in, norm_out = [], []
    handles = []

    def make_layer_hook(k):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            block_out[k].append(h[0, -1].detach().clone())
        return hook

    try:
        for k in range(1, NUM_BLOCKS + 1):
            handles.append(layers[k - 1].register_forward_hook(make_layer_hook(k)))
        handles.append(norm.register_forward_pre_hook(lambda _m, inp: norm_in.append(inp[0][0, -1].detach().clone())))
        handles.append(norm.register_forward_hook(lambda _m, _i, out: norm_out.append(out[0, -1].detach().clone())))
        hooked = _predict(fork, model, processor, unnorm_key, image_uint8, task_label)
    finally:
        for h in handles:
            h.remove()
    hooked_tokens = hooked["sequences"][0, -ACTION_DIM:].cpu().tolist()

    # --- structure (handoff T5) --------------------------------------------------------------
    hs = hooked["hidden_states"]
    assert len(hs) == ACTION_DIM
    assert all(len(s) == NUM_BLOCKS + 1 for s in hs)
    assert all(t.shape[-1] == 4096 for s in hs for t in s)
    assert hs[0][0].shape[0] == 1 and hs[0][0].shape[1] > 1
    assert all(s[0].shape == (1, 1, 4096) for s in hs[1:])  # KV cache: one new token per step

    # each hook fires exactly once per generation step
    for k in range(1, NUM_BLOCKS + 1):
        assert len(block_out[k]) == ACTION_DIM, (k, len(block_out[k]))
    assert len(norm_in) == len(norm_out) == ACTION_DIM

    report = {"attn_implementation": ATTN_IMPL, "unnorm_key": unnorm_key, "task_label": task_label,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "tokens_ref": ref_tokens, "tokens_hooked": hooked_tokens,
              "tokens_ref_no_output_hidden_states": ref_nohs_tokens,
              "block_max_abs_diff": {}, "norm": []}

    # --- check 1: blocks 1..31 exact vs hidden_states[j][k] ----------------------------------
    for k in range(1, NUM_BLOCKS):
        worst = 0.0
        for j in range(ACTION_DIM):
            a = hs[j][k][0, -1]
            b = block_out[k][j]
            worst = max(worst, (a.float() - b.float()).abs().max().item())
            assert torch.equal(a, b), f"block {k} step {j}: max abs diff {worst}"
        report["block_max_abs_diff"][f"block_{k}"] = worst

    # --- check 2: final norm ----------------------------------------------------------------
    atol = float(os.environ.get("VLA_NORM_ATOL", "0.0"))
    eps = norm.variance_epsilon
    for j in range(ACTION_DIM):
        final_j = hs[j][NUM_BLOCKS][0, -1]
        # (a) the norm module's own output on that token is exactly what hidden_states[32] stores
        assert torch.equal(norm_out[j], final_j), f"norm hook output != hidden_states[{j}][32]"
        # (b) pre-hook raw block-32 input -> re-apply the model's own norm (per token, before pooling)
        raw32 = norm_in[j]
        assert torch.equal(raw32, block_out[NUM_BLOCKS][j]), "norm pre-hook input != layers[31] output"
        recomputed = norm(raw32[None, None, :])[0, 0]
        diff = (recomputed.float() - final_j.float()).abs()
        rel = diff / final_j.float().abs().clamp_min(1e-12)
        entry = {"step": j, "eps": eps, "max_abs": diff.max().item(), "max_rel": rel.max().item(),
                 "rel_l2": (diff.norm() / final_j.float().norm()).item(),
                 "norm_ratio": (recomputed.float().norm() / final_j.float().norm()).item()}
        report["norm"].append(entry)
    path = os.environ.get("VLA_T6_REPORT", "t6_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["norm"], indent=1))
    for e in report["norm"]:
        assert e["max_abs"] <= atol, f"RMSNorm(raw block_32) mismatch at step {e['step']}: {e} (VLA_NORM_ATOL={atol})"

    # --- check 3: hooks do not change greedy tokens ------------------------------------------
    assert hooked_tokens == ref_tokens, (ref_tokens, hooked_tokens)
    if ref_nohs_tokens != ref_tokens:  # informational: output_hidden_states flag itself
        print("WARNING: tokens differ with output_hidden_states=False:", ref_nohs_tokens, ref_tokens)
