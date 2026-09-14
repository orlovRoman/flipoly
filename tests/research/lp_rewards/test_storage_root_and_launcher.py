import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import pytest

from polyflip.research.lp_rewards.protocol import (
    DEFAULT_LINUX_STORAGE_FALLBACK,
    compute_file_sha256,
    load_protocol,
    resolve_storage_root,
)

collector_03 = importlib.import_module("scripts.research.lp_rewards.03_run_shadow_collector")
audit_04 = importlib.import_module("scripts.research.lp_rewards.04_audit_data_quality")
eval_05 = importlib.import_module("scripts.research.lp_rewards.05_evaluate_gate_a")
fetch_01 = importlib.import_module("scripts.research.lp_rewards.01_fetch_universe")
allocate_02 = importlib.import_module("scripts.research.lp_rewards.02_rank_and_allocate")


def test_resolve_storage_root_windows_default():
    win_path = r"D:\flipoly-research\lp-rewards"
    resolved = resolve_storage_root(win_path, platform="win32")
    assert resolved == win_path


def test_resolve_storage_root_linux_fallback_for_windows_path(caplog):
    win_path = r"D:\flipoly-research\lp-rewards"
    with caplog.at_level(logging.WARNING):
        resolved = resolve_storage_root(win_path, platform="linux")
    expected = os.path.expanduser(DEFAULT_LINUX_STORAGE_FALLBACK)
    assert resolved == expected
    assert "Windows storage path" in caplog.text
    assert "detected on non-Windows platform 'linux'" in caplog.text


def test_resolve_storage_root_darwin_fallback_for_windows_path(caplog):
    win_path = r"C:\flipoly-research\lp-rewards"
    with caplog.at_level(logging.WARNING):
        resolved = resolve_storage_root(win_path, platform="darwin")
    expected = os.path.expanduser(DEFAULT_LINUX_STORAGE_FALLBACK)
    assert resolved == expected
    assert "Windows storage path" in caplog.text
    assert "detected on non-Windows platform 'darwin'" in caplog.text


def test_resolve_storage_root_linux_valid_unix_path_preserved():
    unix_path = "/var/data/flipoly-research/lp-rewards"
    resolved = resolve_storage_root(unix_path, platform="linux")
    assert resolved == unix_path


def test_resolve_storage_root_env_overrides(monkeypatch):
    win_path = r"D:\flipoly-research\lp-rewards"

    # 1. LP_STORAGE_ROOT override
    monkeypatch.setenv("LP_STORAGE_ROOT", "/custom/storage/root")
    monkeypatch.delenv("LP_STORAGE_PATH", raising=False)
    assert resolve_storage_root(win_path, platform="linux") == "/custom/storage/root"
    assert resolve_storage_root(win_path, platform="win32") == "/custom/storage/root"

    # 2. LP_STORAGE_PATH fallback
    monkeypatch.delenv("LP_STORAGE_ROOT", raising=False)
    monkeypatch.setenv("LP_STORAGE_PATH", "/custom/storage/path")
    assert resolve_storage_root(win_path, platform="linux") == "/custom/storage/path"
    assert resolve_storage_root(win_path, platform="win32") == "/custom/storage/path"

    # 3. LP_STORAGE_ROOT takes precedence over LP_STORAGE_PATH
    monkeypatch.setenv("LP_STORAGE_ROOT", "/priority/root")
    monkeypatch.setenv("LP_STORAGE_PATH", "/secondary/path")
    assert resolve_storage_root(win_path, platform="linux") == "/priority/root"


def test_resolve_storage_root_explicit_override_precedence(monkeypatch):
    win_path = r"D:\flipoly-research\lp-rewards"
    monkeypatch.setenv("LP_STORAGE_ROOT", "/env/storage/root")
    resolved = resolve_storage_root(win_path, override="/explicit/override/path", platform="linux")
    assert resolved == "/explicit/override/path"


def test_resolve_storage_root_whitespace_env_fallback(monkeypatch):
    win_path = r"D:\flipoly-research\lp-rewards"
    monkeypatch.setenv("LP_STORAGE_ROOT", "   ")
    monkeypatch.setenv("LP_STORAGE_PATH", "/fallback/storage/path")
    resolved = resolve_storage_root(win_path, platform="linux")
    assert resolved == "/fallback/storage/path"


def test_resolve_storage_root_tilde_expansion(monkeypatch):
    win_path = r"D:\flipoly-research\lp-rewards"

    # 1. Override with tilde
    resolved = resolve_storage_root(win_path, override="~/my_research/lp_rewards")
    assert resolved == os.path.expanduser("~/my_research/lp_rewards")
    assert not resolved.startswith("~")

    # 2. Env var with tilde
    monkeypatch.setenv("LP_STORAGE_ROOT", "~/env_research/lp_rewards")
    monkeypatch.delenv("LP_STORAGE_PATH", raising=False)
    resolved_env = resolve_storage_root(win_path)
    assert resolved_env == os.path.expanduser("~/env_research/lp_rewards")
    assert not resolved_env.startswith("~")


def test_load_protocol_with_env_storage_root(monkeypatch):
    custom_root = "/mnt/fast_disk/flipoly/lp-rewards"
    monkeypatch.setenv("LP_STORAGE_ROOT", custom_root)

    protocol = load_protocol()
    assert protocol.data_storage.root_path == custom_root
    # SHA-256 hash must remain strictly verified from the YAML file
    assert protocol.sha256_hash is not None
    base = Path(__file__).resolve().parents[3]
    protocol_yaml = base / "artifacts" / "research" / "lp_rewards" / "protocol_v0.1.yaml"
    assert protocol.sha256_hash == compute_file_sha256(protocol_yaml)


def test_load_protocol_with_explicit_storage_root(tmp_path):
    protocol = load_protocol(storage_root=str(tmp_path))
    assert protocol.data_storage.root_path == str(tmp_path)


def test_script_03_parser_supports_storage_root():
    # Directly test the real build_parser defined in 03_run_shadow_collector
    parser = collector_03.build_parser()
    args = parser.parse_args(["--storage-root", "/custom/shadow/path", "--smoke-seconds", "5", "--log-file", "/tmp/shadow.log"])
    assert args.storage_root == "/custom/shadow/path"
    assert args.smoke_seconds == 5.0
    assert args.log_file == "/tmp/shadow.log"


@pytest.mark.asyncio
async def test_script_03_main_with_storage_root_argv(tmp_path):
    # Test that 03_run_shadow_collector main accepts argv and parses --storage-root and --smoke-seconds
    await collector_03.main(argv=["--storage-root", str(tmp_path), "--smoke-seconds", "0.01"])
    # Storage directory must have been created
    assert tmp_path.exists()


def test_script_04_cli_storage_root(tmp_path, capsys):
    # Test that 04_audit_data_quality main accepts --storage-root and runs
    # (Exits normally with PENDING_DATA_ACCUMULATION when empty, pointing to tmp_path)
    audit_04.main(argv=["--storage-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert f"Storage Root: {tmp_path}" in captured.out
    assert "Overall Audit Status: PENDING_DATA_ACCUMULATION" in captured.out


def test_script_05_cli_storage_root(tmp_path, capsys):
    # Test that 05_evaluate_gate_a main accepts --storage-root and points to tmp_path
    eval_05.main(argv=["--storage-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert f"No completed daily evaluations found in {tmp_path / 'daily_evaluations'}" in captured.out


def test_script_01_and_02_cli_storage_root(tmp_path, capsys):
    # 01_fetch_universe build_parser test
    parser_01 = fetch_01.build_parser()
    args_01 = parser_01.parse_args(["--storage-root", str(tmp_path)])
    assert args_01.storage_root == str(tmp_path)

    # 02_rank_and_allocate should exit 1 when universe_active.json not found in tmp_path
    with pytest.raises(SystemExit) as exc_info:
        allocate_02.main(argv=["--storage-root", str(tmp_path)])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "universe_active.json not found" in captured.out


def test_run_continuous_shadow_script_integrity():
    repo_root = Path(__file__).resolve().parents[3]
    sh_script = repo_root / "scripts" / "research" / "lp_rewards" / "run_continuous_shadow.sh"

    assert sh_script.exists()
    content = sh_script.read_text(encoding="utf-8")

    # 1. No bare python invocation
    assert "exec python -u" not in content
    assert 'exec "${PYTHON_CMD[@]}" -u' in content

    # 2. Key interpreter detection targets present
    assert "poetry run python" in content
    assert "${HOME:-}/.local/bin/poetry" in content
    assert "~/.local/bin/poetry" in content
    assert "uv run python" in content
    assert "${HOME:-}/.local/bin/uv" in content
    assert "~/.local/bin/uv" in content
    assert "${HOME:-}/.cargo/bin/uv" in content
    assert "~/.cargo/bin/uv" in content
    assert "${VIRTUAL_ENV}/bin/python" in content
    assert "${REPO_ROOT}/.venv/bin/python" in content
    assert "/usr/bin/python3" in content
    assert "python3" in content

    # 3. PATH and storage root resolution
    assert 'export PATH="${HOME:-~}/.local/bin' in content
    assert 'DEFAULT_STORAGE_ROOT="${HOME:-~}/flipoly-research/lp-rewards"' in content
    assert 'export LP_STORAGE_ROOT="${LP_STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"' in content
    assert "--storage-root" in content


def test_run_continuous_shadow_git_executable_mode():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "ls-files", "--stage", "scripts/research/lp_rewards/run_continuous_shadow.sh"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    # Git stage mode must be 100755 (executable file)
    assert result.stdout.startswith("100755")


def test_run_continuous_shadow_bash_live_enabled_blocked():
    import shutil
    if not shutil.which("bash"):
        pytest.skip("bash not available on this platform")

    repo_root = Path(__file__).resolve().parents[3]
    # Pass LP_LIVE_ENABLED=true inside bash -c command to guarantee environment delivery
    cmd = ["bash", "-c", "LP_LIVE_ENABLED=true scripts/research/lp_rewards/run_continuous_shadow.sh"]
    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert result.returncode == 1
    assert "[FATAL] LP_LIVE_ENABLED is set to 'true'" in result.stderr


def test_run_continuous_shadow_bash_execution_echo_runner():
    import shutil
    if not shutil.which("bash"):
        pytest.skip("bash not available on this platform")

    repo_root = Path(__file__).resolve().parents[3]
    # Test execution with PYTHON_RUNNER=echo and CLI argument overrides
    cmd = [
        "bash",
        "-c",
        'PYTHON_RUNNER="echo" scripts/research/lp_rewards/run_continuous_shadow.sh '
        '--storage-root /custom/storage/path --log-file /custom/log/path.log --smoke-seconds 5',
    ]
    result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert result.returncode == 0
    assert "Storage Root: /custom/storage/path" in result.stdout
    assert "Log file: /custom/log/path.log" in result.stdout
    # The echoed command line must contain the arguments without duplicates
    assert "--storage-root /custom/storage/path" in result.stdout
    assert "--log-file /custom/log/path.log" in result.stdout
    assert result.stdout.count("--storage-root") == 1  # exactly 1 in echoed command, no duplicate
    assert result.stdout.count("--log-file") == 1  # exactly 1 in echoed command, no duplicate
