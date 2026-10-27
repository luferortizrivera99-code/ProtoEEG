import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from protopnet.eeg_utilities.sleep_edf_dataset import SleepEDFDataset, split_by_subject


def test_loading_and_subject_split(tmp_path):
    # Dos noches del mismo sujeto deben permanecer juntas.
    original = np.arange(6 * 3 * 3000, dtype=np.float32).reshape(6, 3, 3000)
    for name, subject in [("night1", "s1"), ("night2", "s1"), ("night3", "s2"), ("night4", "s3")]:
        subj = np.array(subject) if name != "night4" else np.repeat(subject, 6)
        np.savez(tmp_path / f"{name}.npz", X=original, y=np.arange(1, 7), subj=subj)
    dataset = SleepEDFDataset(tmp_path)
    assert len(dataset) == 24
    np.testing.assert_array_equal(dataset[0]["img"].numpy(), original[0])
    assert dataset.y.tolist() == list(range(6)) * 4
    assert len(set(dataset.sample_ids)) == 24
    train, val, test = split_by_subject(dataset, ["s1"], ["s2"], ["s3"])
    assert [len(part) for part in (train, val, test)] == [12, 6, 6]
    batch = next(iter(DataLoader(train, batch_size=6, shuffle=False)))
    assert batch["img"].shape == (6, 3, 3000)
    assert batch["img"].dtype == torch.float32
    assert batch["target"].dtype == torch.int64
    assert batch["target"].tolist() == list(range(6))
    assert set(batch["subject_id"]) == {"s1"}
    with pytest.raises(ValueError, match="más de una"):
        split_by_subject(dataset, ["s1"], ["s1", "s2"], ["s3"])
    with pytest.raises(ValueError, match="sin asignar"):
        split_by_subject(dataset, ["s1"], ["s2"], ["other"])


@pytest.mark.parametrize("problem", ["shape", "label", "length", "nan", "subject"])
def test_invalid_file_is_rejected(tmp_path, problem):
    x = np.zeros((2, 3, 3000), dtype=np.float32)
    y = np.array([1, 6])
    subj = np.array("s1")
    if problem == "shape":
        x = x[:, :2]
    elif problem == "label":
        y[0] = 0
    elif problem == "length":
        y = y[:1]
    elif problem == "nan":
        x[0, 0, 0] = np.nan
    elif problem == "subject":
        subj = np.array(["s1", "s2", "s3"])
    np.savez(tmp_path / "bad.npz", X=x, y=y, subj=subj)
    with pytest.raises(ValueError):
        SleepEDFDataset(tmp_path)
