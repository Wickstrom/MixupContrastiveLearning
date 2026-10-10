import csv

import numpy as np
import pytest

import mcl.cli
from mcl.cli import _build_parser, _save_history, main


def test_parser_defaults():
    args = _build_parser().parse_args(["train"])
    assert args.command == "train"
    assert args.dataset == "GunPoint"
    assert args.epochs == 200
    assert args.alpha == 1.0
    assert args.lr == 1e-3
    assert args.tau == 0.5
    assert args.batch_size == 0
    assert args.device is None
    assert args.seed is None
    assert args.save_history is None


def test_save_history_round_trip(tmp_path):
    path = tmp_path / "history.csv"
    history = {"epoch_losses": [1.0, 0.5], "accuracies": [0.5, 0.9]}
    _save_history(history, str(path))

    rows = list(csv.reader(path.open()))
    assert rows[0] == ["epoch", "loss", "accuracy"]
    assert rows[1] == ["0", "1.0", "0.5"]
    assert rows[2] == ["1", "0.5", "0.9"]


def test_save_history_without_accuracies(tmp_path):
    path = tmp_path / "history.csv"
    _save_history({"epoch_losses": [1.0], "accuracies": []}, str(path))

    rows = list(csv.reader(path.open()))
    assert rows[1] == ["0", "1.0", ""]


def test_save_history_reports_write_error(tmp_path):
    with pytest.raises(SystemExit):
        _save_history(
            {"epoch_losses": [1.0], "accuracies": []},
            str(tmp_path / "missing" / "history.csv"),
        )


def test_main_runs_train_command(monkeypatch, tmp_path, capsys):
    x = np.zeros((4, 1, 10), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    monkeypatch.setattr(mcl.cli, "load_dataset", lambda name, split: (x, y))

    calls = {}

    def fake_train(model, train_set, test_set, **kwargs):
        calls.update(kwargs)
        return {"losses": [0.5], "epoch_losses": [0.5], "accuracies": [0.75]}

    monkeypatch.setattr(mcl.cli, "train", fake_train)

    path = tmp_path / "history.csv"
    main(["train", "--epochs", "1", "--seed", "0", "--save-history", str(path)])

    assert calls["epochs"] == 1
    assert calls["batch_size"] is None
    out = capsys.readouterr().out
    assert "Final 1-NN test accuracy: 0.7500" in out
    assert path.exists()
