import pytest

torch = pytest.importorskip("torch")

from robot_arm_il.config import ACTION_DIM, STATE_DIM  # noqa: E402
from robot_arm_il.policies import ACTPolicy, MLPPolicy, build_policy  # noqa: E402


def batch(b=8, k=20):
    obs = torch.randn(b, STATE_DIM)
    act = torch.randn(b, k, ACTION_DIM)
    mask = torch.ones(b, k, dtype=torch.bool)
    return obs, act, mask


def test_mlp_shapes_and_loss():
    model = MLPPolicy()
    obs, act, mask = batch(k=1)
    assert model(obs).shape == (8, 1, ACTION_DIM)
    out = model.loss(obs, act, mask)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


def test_act_shapes_and_loss():
    model = ACTPolicy(chunk=10, d_model=32, heads=2, latent=4)
    obs, act, mask = batch(k=10)
    assert model(obs).shape == (8, 10, ACTION_DIM)
    out = model.loss(obs, act, mask)
    out["loss"].backward()
    assert out["kl"] >= 0


def test_act_masked_targets_do_not_change_the_loss():
    torch.manual_seed(0)
    model = ACTPolicy(chunk=6, d_model=32, heads=2, latent=4, dropout=0.0).eval()
    obs, act, mask = batch(b=4, k=6)
    mask[:, 4:] = False
    other = act.clone()
    other[:, 4:] = 1e3  # garbage in the padded region
    torch.manual_seed(1)
    a = model.loss(obs, act, mask)["l1"]
    torch.manual_seed(1)
    b = model.loss(obs, other, mask)["l1"]
    assert torch.isclose(a, b)


def test_inference_is_deterministic():
    model = ACTPolicy(chunk=6, d_model=32, heads=2, latent=4).eval()
    obs = torch.randn(3, STATE_DIM)
    assert torch.equal(model(obs), model(obs))


def test_build_policy_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_policy("diffusion")
    assert isinstance(build_policy("bc"), MLPPolicy)


def test_policy_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    model = MLPPolicy(hidden=64)
    obs, act, mask = batch(b=16, k=1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(model.loss(obs, act, mask)["l1"])
    for _ in range(300):
        opt.zero_grad()
        model.loss(obs, act, mask)["loss"].backward()
        opt.step()
    assert float(model.loss(obs, act, mask)["l1"]) < 0.2 * first



def test_mlp_with_history_and_chunk_masks_padded_steps():
    model = MLPPolicy(history=3, chunk=4)
    obs = torch.randn(2, STATE_DIM * 3)
    act = torch.zeros(2, 4, ACTION_DIM)
    assert model(obs).shape == (2, 4, ACTION_DIM)
    mask = torch.tensor([[True, True, False, False]] * 2)
    pred = model(obs)
    expected = pred[:, :2].abs().mean()
    assert torch.isclose(model.loss(obs, act, mask)["loss"], expected)


def test_latent_gap_reports_both_errors():
    from robot_arm_il.latent import latent_gap
    from robot_arm_il.policies import ACTPolicy

    model = ACTPolicy(chunk=4)
    obs, actions = torch.randn(6, 13), torch.randn(6, 8, 6)
    result = latent_gap(model, obs, actions, torch.ones(6, 8, dtype=torch.bool))
    assert result["l1_zero"] > 0 and result["l1_posterior"] > 0 and result["kl"] >= 0
