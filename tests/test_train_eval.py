import json

import pytest

pytest.importorskip("torch")

from robot_arm_il.collect import main as collect_main  # noqa: E402
from robot_arm_il.policy_eval import PolicyRunner  # noqa: E402
from robot_arm_il.policy_eval import main as eval_main  # noqa: E402
from robot_arm_il.train import main as train_main  # noqa: E402


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "demos.npz"
    collect_main(["--episodes", "12", "--out", str(path)])
    return path


@pytest.mark.parametrize("kind", ["bc", "act"])
def test_train_writes_checkpoint_and_eval_runs(kind, demos, tmp_path, capsys):
    ckpt = tmp_path / f"{kind}.pt"
    train_main(["--policy", kind, "--demos", str(demos), "--epochs", "1", "--out", str(ckpt)])
    log = json.loads(ckpt.with_suffix(".json").read_text())
    assert log["policy"] == kind and log["best_val_l1"] > 0
    out = tmp_path / "eval.json"
    eval_main(["--ckpt", str(ckpt), "--episodes", "2", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["episodes"] == 2 and len(res["episodes_detail"]) == 2
    assert 0.0 <= res["success_rate"] <= 1.0


def test_temporal_ensemble_with_equal_weights_averages_overlapping_chunks(demos, tmp_path):
    ckpt = tmp_path / "act.pt"
    train_main(["--policy", "act", "--demos", str(demos), "--epochs", "1", "--out", str(ckpt)])
    runner = PolicyRunner(str(ckpt), ensemble=0.0)
    obs = __import__("numpy").zeros(13, dtype="float32")
    first = runner.act(obs)
    assert first.shape == (6,)
    assert len(runner.pending) == runner.chunk - 1  # steps 1..K-1 wait for later chunks
    runner.act(obs)
    assert len(runner.pending[2]) == 2  # step 2 now holds predictions from two chunks


def test_train_subset_uses_requested_number_of_episodes(demos, tmp_path):
    ckpt = tmp_path / "bc.pt"
    train_main(
        ["--policy", "bc", "--demos", str(demos), "--epochs", "1", "--n-train-episodes", "3",
         "--out", str(ckpt)]
    )
    assert json.loads(ckpt.with_suffix(".json").read_text())["train_episodes"] == 3
