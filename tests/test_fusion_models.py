"""F1 (concat_fusion) and F2 (static_weight_fusion): contract, masks, parameters, edge cases. CPU only."""
import pytest
import torch

from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.trainer import collate_records, train_model
from failure_prob.models import get_model, list_models
from failure_prob.models.static_weight_fusion import entropy_of

K, D = 3, 6


def batch(n=4, k=K, d=D, seed=0):
    return collate_records(make_synthetic_records(n=n, k=k, d=d, seed=seed))


def n_params(m, trainable=True):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)


@pytest.mark.parametrize("name", ["concat_fusion", "static_weight_fusion"])
@pytest.mark.parametrize("head", ["mlp", "linear"])
def test_forward_backward_shapes(name, head):
    b = batch()
    m = get_model(name, input_dim=D, num_layers=K, hidden_dim=5, head=head)
    out = m(b)
    assert out.failure_logits.shape == b["mask"].shape and out.prediction_level == "timestep"
    m.loss(b).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_registered():
    assert {"concat_fusion", "static_weight_fusion"} <= set(list_models())


def test_concat_default_unchanged_and_linear_param_count():
    torch.manual_seed(0)
    m = get_model("concat_fusion", input_dim=D, num_layers=K, hidden_dim=7)
    assert n_params(m) == (K * D) * 7 + 7 + 7 + 1  # original MLPHead on K*D inputs
    assert n_params(get_model("concat_fusion", input_dim=D, num_layers=K, head="linear")) == K * D + 1


def test_static_weights_are_a_distribution_and_parameter_matched():
    m = get_model("static_weight_fusion", input_dim=D, num_layers=K, hidden_dim=7)
    single = get_model("single_layer_mlp", input_dim=D, num_layers=1, hidden_dim=7)
    assert n_params(m) == n_params(single) + K + 1  # head identical to one-layer probe; + K logits + gamma
    w = m(batch()).auxiliary["layer_weights"]
    assert w.shape == (K,) and torch.allclose(w.sum(), torch.tensor(1.0)) and (w >= 0).all()
    assert torch.allclose(w, torch.full((K,), 1 / K))  # uniform init
    assert entropy_of(w) == pytest.approx(torch.log(torch.tensor(float(K))).item())


def test_static_weights_learn_to_pick_the_informative_layer():
    torch.manual_seed(0)
    recs = make_synthetic_records(n=24, k=K, d=D, seed=3)
    for r in recs:  # layer 1 carries the label, layers 0 and 2 are noise
        r["features"][:, :, 1] = (r["labels"].item() * 2 - 1) * 3.0
    m = get_model("static_weight_fusion", input_dim=D, num_layers=K, head="linear")
    train_model(m, recs, epochs=40, lr=5e-2)
    w = m.layer_weights()
    assert w.argmax().item() == 1


def test_fixed_weights_buffer_not_trainable_and_equal_to_mean():
    m = get_model("static_weight_fusion", input_dim=D, num_layers=K, head="linear", learn_weights=False)
    assert "fixed_weights" in dict(m.named_buffers()) and n_params(m) == D + 1
    b = batch()
    with torch.no_grad():
        expect = m.head(b["features"].mean(2))
    assert torch.allclose(m(b).failure_logits, expect, atol=1e-6)
    custom = get_model("static_weight_fusion", input_dim=D, num_layers=K, learn_weights=False, init_weights=[0, 0, 2])
    assert torch.equal(custom.layer_weights(), torch.tensor([0.0, 0.0, 1.0]))


def test_learnable_init_weights_start_at_the_given_mix():
    m = get_model("static_weight_fusion", input_dim=D, num_layers=K, init_weights=[1, 2, 1])
    assert torch.allclose(m.layer_weights(), torch.tensor([0.25, 0.5, 0.25]))


@pytest.mark.parametrize("init, learn, msg", [
    ([1, 1], True, "needs 3 values"),
    ([1, -1, 1], True, "non-negative"),
    ([0, 0, 0], False, "all be zero"),
    ([0, 1, 1], True, "strictly positive"),
    ([1, float("nan"), 1], True, "finite"),
])
def test_bad_init_weights(init, learn, msg):
    with pytest.raises(ValueError, match=msg):
        get_model("static_weight_fusion", input_dim=D, num_layers=K, init_weights=init, learn_weights=learn)


@pytest.mark.parametrize("name", ["concat_fusion", "static_weight_fusion"])
def test_wrong_layer_count_or_dim_fails_readably(name):
    m = get_model(name, input_dim=D, num_layers=K)
    with pytest.raises(ValueError, match="Expected 3 layers"):
        m(batch(k=2))
    with pytest.raises(ValueError, match="Expected feature dim 6"):
        m(batch(d=D + 1))
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        get_model(name, input_dim=D, num_layers=0)
    with pytest.raises(ValueError, match="Unknown probe head"):
        get_model(name, input_dim=D, num_layers=K, head="transformer")


@pytest.mark.parametrize("name", ["concat_fusion", "static_weight_fusion"])
def test_single_layer_and_3d_input(name):
    b = batch(k=1)
    m = get_model(name, input_dim=D, num_layers=1)
    full = m(b).failure_logits
    b3 = dict(b, features=b["features"][:, :, 0])  # [B,T,D] is accepted as one layer
    assert torch.allclose(m(b3).failure_logits, full)


@pytest.mark.parametrize("name", ["concat_fusion", "static_weight_fusion"])
def test_padding_does_not_change_valid_logits_or_loss(name):
    b = batch(n=5)
    assert not b["mask"].all()  # synthetic records have different lengths -> padding exists
    m = get_model(name, input_dim=D, num_layers=K)
    m.eval()
    noisy = dict(b, features=b["features"].clone())
    noisy["features"][~b["mask"]] = 1e3  # garbage in padded steps
    with torch.no_grad():
        assert torch.allclose(m(b).failure_logits[b["mask"]], m(noisy).failure_logits[b["mask"]])
        assert torch.allclose(m.loss(b), m.loss(noisy))


@pytest.mark.parametrize("name", ["concat_fusion", "static_weight_fusion"])
def test_state_dict_roundtrip(name, tmp_path):
    torch.manual_seed(1)
    m = get_model(name, input_dim=D, num_layers=K, init_weights=[1, 2, 3])
    train_model(m, make_synthetic_records(n=6, k=K, d=D), epochs=1)
    torch.save(m.state_dict(), tmp_path / "m.pt")
    m2 = get_model(name, input_dim=D, num_layers=K)
    m2.load_state_dict(torch.load(tmp_path / "m.pt"))
    b = batch()
    m.eval(), m2.eval()
    assert torch.allclose(m(b).failure_logits, m2(b).failure_logits)


# --------------------------------------------------------------------------- F3: projection to a shared dimension

def test_projection_to_shared_dim_shapes_and_parameter_match_with_concat():
    d, k, p = 48, 3, 16
    m = get_model("static_weight_fusion", input_dim=d, num_layers=k, hidden_dim=p, proj_dim=p)
    b = batch(k=k, d=d)
    out = m(b)
    assert out.failure_logits.shape == b["mask"].shape and torch.isfinite(out.failure_logits).all()
    out.failure_logits.sum().backward()
    assert all(lin.weight.grad is not None for lin in m.proj)
    concat = get_model("concat_fusion", input_dim=d, num_layers=k, hidden_dim=p)
    # projections K*(D*P+P) + head(P->P->1) vs concat head (K*D->P->1): equal up to the small P*P + K*P terms
    assert abs(n_params(m) - n_params(concat)) <= p * p + k * p + k + 1
    assert m.layer_weights().shape == (k,)


def test_projection_layers_are_independent_per_layer():
    m = get_model("static_weight_fusion", input_dim=8, num_layers=2, hidden_dim=4, proj_dim=4)
    assert m.proj[0].weight.data_ptr() != m.proj[1].weight.data_ptr()


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_bad_proj_dim(bad):
    with pytest.raises(ValueError, match="proj_dim"):
        get_model("static_weight_fusion", input_dim=8, num_layers=2, proj_dim=bad)


def test_projection_with_fixed_uniform_weights_and_k1():
    m = get_model("static_weight_fusion", input_dim=8, num_layers=1, hidden_dim=4, proj_dim=4, learn_weights=False)
    b = batch(k=1, d=8)
    assert m(b).failure_logits.shape == b["mask"].shape


def test_fixed_weights_is_the_s5_s6_feature_average_baseline():
    """S5/S6 definition (Rhea): z-score each layer with train-only stats (the runner/loader does this), AVERAGE THE
    FEATURES, then ONE probe. static_weight_fusion --fixed-weights (no --proj-dim) must be exactly that: no score
    averaging, no weights after the probe, nothing trainable outside the probe head."""
    import torch
    from failure_prob.models import get_model
    m = get_model("static_weight_fusion", input_dim=8, num_layers=3, hidden_dim=4, learn_weights=False).eval()
    x = torch.randn(2, 5, 3, 8)
    out = m({"features": x, "mask": torch.ones(2, 5, dtype=torch.bool), "labels": torch.zeros(2)}).failure_logits
    ref = m.head(x.mean(dim=2))
    assert torch.allclose(out.reshape(ref.shape), ref, atol=1e-6)
    assert not [n for n, p in m.named_parameters() if p.requires_grad and not n.startswith("head")]
