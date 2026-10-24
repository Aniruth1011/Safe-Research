from failure_prob.data.feature_store import FeatureMetadata, FeatureStore
from failure_prob.data.synthetic import make_synthetic_records
from failure_prob.engine.trainer import collate_records, train_model
from failure_prob.models import get_model, list_models

def test_registry_and_forward_backward():
    assert {"safe_mlp", "safe_lstm", "single_layer_mlp", "concat_fusion", "gated_fusion", "residual_fusion"} <= set(list_models())
    batch=collate_records(make_synthetic_records(n=3,k=2,d=8)); model=get_model("gated_fusion", input_dim=8, num_layers=2)
    assert model(batch).failure_logits.shape == batch["mask"].shape
    model.loss(batch).backward()

def test_all_models_cpu():
    batch=collate_records(make_synthetic_records(n=4,k=3,d=5))
    for name in list_models():
        model=get_model(name, input_dim=5, num_layers=3)
        assert model(batch).failure_logits.shape == batch["mask"].shape

def test_train_smoke_and_store(tmp_path):
    records=make_synthetic_records(n=6,k=2,d=4); model=get_model("concat_fusion", input_dim=4, num_layers=2); assert len(train_model(model, records, epochs=1)) == 1
    store=FeatureStore(tmp_path / "features"); store.save(records, FeatureMetadata("openvla","test","libero10","v1",(8,16),num_layers=32)); assert store.load([8])[0]["features"].shape[2] == 1

def test_missing_layer_fails(tmp_path):
    store=FeatureStore(tmp_path / "features"); store.save(make_synthetic_records(n=1,k=1), FeatureMetadata("openvla","test","libero10","v1",(8,),num_layers=32))
    try: store.load([16])
    except ValueError as exc: assert "missing" in str(exc)
    else: raise AssertionError("missing layers must fail")
