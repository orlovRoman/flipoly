
import pytest

from polyflip.research.canonical_models.guards import (
    assert_local_workdir, assert_train_allowed, configure_thread_limits,
)


def test_sshfs_refused(tmp_path):
    with pytest.raises(RuntimeError, match="SSHFS"):
        assert_local_workdir("//sshfs/server/share")
    with pytest.raises(RuntimeError, match="SSHFS"):
        assert_local_workdir("\\\\sshfs\\share")


def test_local_passes_and_caps(tmp_path):
    p = assert_train_allowed(str(tmp_path))
    assert p.exists()
    vals = configure_thread_limits(4)
    assert vals["OMP_NUM_THREADS"] == "4"


def test_server_role_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYFLIP_ROLE", "server")
    with pytest.raises(RuntimeError, match="server"):
        assert_train_allowed(str(tmp_path))
    monkeypatch.setenv("POLYFLIP_ROLE", "local")


def test_train_refuses_sshfs():
    with pytest.raises(RuntimeError):
        assert_train_allowed("//sshfs/box/research")
