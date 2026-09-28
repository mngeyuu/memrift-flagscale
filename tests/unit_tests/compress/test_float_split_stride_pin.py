import types

import pytest

from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp


class MergeResult(str):
    pass


def test_merge_copy_env_uses_extension_merge_copy(monkeypatch):
    calls = []

    fake_ext = types.SimpleNamespace(
        split=lambda *args: None,
        split_copy=lambda *args: None,
        merge=lambda *args: calls.append(("merge", args)) or "mapped",
        merge_copy=lambda *args: calls.append(("merge_copy", args)) or (MergeResult("copy"), "staging"),
        acquire_pin=lambda *args: None,
        release_pin=lambda *args: None,
        release_cuda=lambda *args: None,
    )

    monkeypatch.setattr(fs_sp, "_AVAILABLE", True)
    monkeypatch.setattr(fs_sp, "_ext", fake_ext)
    monkeypatch.setenv("MEMRIFT_MERGE_PATH", "copy")

    result = fs_sp.merge("exp", "sm", [2], [1], 0, "dtype", 123)

    assert result == "copy"
    assert result._memrift_merge_staging_exp == "staging"
    assert calls == [("merge_copy", ("exp", "sm", [2], [1], 0, "dtype", 123))]


def test_merge_rejects_unknown_merge_path(monkeypatch):
    fake_ext = types.SimpleNamespace(
        split=lambda *args: None,
        split_copy=lambda *args: None,
        merge=lambda *args: "mapped",
        merge_copy=lambda *args: (MergeResult("copy"), "staging"),
        acquire_pin=lambda *args: None,
        release_pin=lambda *args: None,
        release_cuda=lambda *args: None,
    )

    monkeypatch.setattr(fs_sp, "_AVAILABLE", True)
    monkeypatch.setattr(fs_sp, "_ext", fake_ext)
    monkeypatch.setenv("MEMRIFT_MERGE_PATH", "bad")

    with pytest.raises(ValueError, match="MEMRIFT_MERGE_PATH"):
        fs_sp.merge("exp", "sm", [2], [1], 0, "dtype", 123)
