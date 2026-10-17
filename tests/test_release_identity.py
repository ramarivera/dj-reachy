"""Offline regressions for the independently installable Remix release."""

import os
import sys
import json
import subprocess
from pathlib import Path

import pytest
from setuptools import find_packages, find_namespace_packages
from setuptools.config.pyprojecttoml import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_distribution_uses_distinct_remix_identity() -> None:
    """Installing the fork must not replace the vanilla distribution."""
    metadata = load_file(PROJECT_ROOT / "pyproject.toml")
    assert metadata["project"]["name"] == "dj-reachy-remix"
    assert metadata["project"]["requires-python"] == ">=3.10"


def test_console_script_does_not_replace_vanilla_command() -> None:
    """Only the independently named console command belongs to this release."""
    metadata = load_file(PROJECT_ROOT / "pyproject.toml")
    assert metadata["project"]["scripts"] == {"dj-reachy-remix": "dj_reachy_remix.main:main"}


def test_app_manager_entry_point_has_independent_slug() -> None:
    """Reachy's app registry must distinguish the fork from vanilla."""
    metadata = load_file(PROJECT_ROOT / "pyproject.toml")
    assert metadata["project"]["entry-points"]["reachy_mini_apps"] == {
        "dj_reachy_remix": "dj_reachy_remix.main:ReachyMiniConversationApp",
    }


def test_package_data_belongs_to_remix_namespace() -> None:
    """Renaming Python modules must not strand prompts or static assets."""
    metadata = load_file(PROJECT_ROOT / "pyproject.toml")
    package_data = metadata["tool"]["setuptools"]["package-data"]
    assert "dj_reachy" not in package_data
    assert {
        "images/*", "static/*", ".env.example", "demos/**/*.txt",
        "prompts_library/*.txt", "profiles/**/*.txt", "prompts/**/*.txt",
    } <= set(package_data["dj_reachy_remix"])


def test_package_discovery_does_not_ship_vanilla_namespace() -> None:
    """Use setuptools discovery, including namespace-only compatibility aliases."""
    metadata = load_file(PROJECT_ROOT / "pyproject.toml")
    options = metadata["tool"]["setuptools"]["packages"]["find"]
    discover = find_namespace_packages if options.get("namespaces", True) else find_packages
    packages = {
        name
        for directory in options.get("where", ["."])
        for name in discover(
            where=str(PROJECT_ROOT / directory),
            include=options.get("include", ["*"]),
            exclude=options.get("exclude", []),
        )
    }
    assert "dj_reachy_remix" in packages
    assert not {name for name in packages if name == "dj_reachy" or name.startswith("dj_reachy.")}
    assert not (PROJECT_ROOT / "src" / "dj_reachy").exists()
    # The protected *internal* profile name is intentionally not the release slug.
    assert "dj_reachy_remix.profiles.dj_reachy" in packages


_CONFIG_PROBE = r'''
import os
import sys
import json
import socket
import importlib
import importlib.abc
from pathlib import Path

source, sandbox, expected_home, expected_cache = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
read_roots = [sandbox, source, Path(sys.prefix), Path(sys.base_prefix)]
read_roots += [Path(item).resolve() for item in sys.path if item and Path(item).is_dir()]


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def audit(event, args):
    if event.startswith("socket.") and event not in {"socket.__new__"}:
        raise RuntimeError("offline config probe blocked networking: " + event)
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
        raise RuntimeError("offline config probe blocked child process")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(value in mode for value in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        if not inside(path, [sandbox] if writing else read_roots):
            raise RuntimeError("offline config probe blocked filesystem escape: " + str(path))
    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
        paths = args[:2] if event == "os.rename" else args[:1]
        for value in paths:
            if isinstance(value, (str, bytes)) and not inside(Path(os.fsdecode(value)).resolve(), [sandbox]):
                raise RuntimeError("offline config probe blocked mutation outside sandbox")


class NoEffectsImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "reachy_mini", "reachy_mini_dances_library", "reachy_mini_toolbox",
            "openai", "elevenlabs", "huggingface_hub", "zenoh", "sounddevice", "cv2",
        }:
            raise RuntimeError("pure config import attempted a device/provider import: " + fullname)
        return None


sys.addaudithook(audit)
sys.meta_path.insert(0, NoEffectsImports())
# Prove this subprocess's guard is active before importing production code.
try:
    socket.getaddrinfo("offline-guard.invalid", 443)
except RuntimeError as exc:
    assert "blocked networking" in str(exc)
else:
    raise AssertionError("offline config guard allowed DNS")
try:
    (sandbox.parent / "forbidden-config-probe-write").write_text("must not write")
except RuntimeError as exc:
    assert "filesystem escape" in str(exc)
else:
    raise AssertionError("offline config guard allowed an escaped write")

module = importlib.import_module("dj_reachy_remix.config")
config = module.config
assert Path(module.__file__).resolve() == source / "dj_reachy_remix" / "config.py"
assert config.APP_HOME == expected_home, (config.APP_HOME, expected_home)
assert config.SONGS_DIR == expected_home / "songs"
assert config.RUNTIME_SETTINGS_PATH == expected_home / "settings.json"
assert module.runtime_settings_store.path == config.RUNTIME_SETTINGS_PATH
assert Path(config.HF_HOME) == expected_cache, (config.HF_HOME, expected_cache)
assert config.DEFAULT_PROFILE == config.REACHY_MINI_CUSTOM_PROFILE == "dj_reachy"
assert config.OPENAI_API_KEY is None and config.ELEVENLABS_API_KEY is None and config.HF_TOKEN is None
assert config.RUNTIME_SETTINGS["music"]["duration_ms"] == 30000, "read vanilla settings"
# Exercise the real persistence API, not a stand-in path resolver.
updated = module.update_runtime_settings({"music": {"duration_ms": 10000}})
assert json.loads(config.RUNTIME_SETTINGS_PATH.read_text()) == updated
assert updated["music"]["duration_ms"] == 10000
assert config.MUSIC_FORCE_INSTRUMENTAL is False
# These are fixture writes to configured destinations, not provider/download calls.
config.SONGS_DIR.mkdir(parents=True, exist_ok=True)
(config.SONGS_DIR / "fixture-song.bin").write_bytes(b"offline-song-fixture")
cache = Path(config.HF_HOME)
cache.mkdir(parents=True, exist_ok=True)
(cache / "fixture-cache.txt").write_text("offline-cache-fixture")
reloaded = importlib.reload(module)
assert reloaded.config.RUNTIME_SETTINGS["music"]["duration_ms"] == 10000
assert reloaded.config.APP_HOME == expected_home
assert "dj_reachy" not in sys.modules
print("REMIX_CONFIG_ISOLATION_OK")
'''


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Include directory membership as well as exact sentinel contents."""
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.mark.parametrize("selection", ["xdg", "home", "explicit", "empty", "expanded", "hf_override"])
def test_config_loader_keeps_remix_writes_out_of_vanilla(selection: str, tmp_path: Path) -> None:
    """Fresh imports own all selectors and never read or mutate vanilla state."""
    roots = {name: tmp_path / name for name in ("home", "data", "config", "state", "cache", "tmp")}
    for root in roots.values():
        root.mkdir()
    # An allowlist, rather than copying os.environ, strips credentials, broker
    # selectors, PYTHONPATH/PYTHONHOME/PYTHONUSERBASE and inherited HF/app homes.
    env = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR", "LANG") if name in os.environ}
    env.update({
        "HOME": str(roots["home"]), "XDG_DATA_HOME": str(roots["data"]),
        "XDG_CONFIG_HOME": str(roots["config"]), "XDG_STATE_HOME": str(roots["state"]),
        "XDG_CACHE_HOME": str(roots["cache"]), "TMPDIR": str(roots["tmp"]),
        "HF_HUB_OFFLINE": "1", "GRADIO_ANALYTICS_ENABLED": "False",
    })
    vanilla_roots = [tmp_path / "inherited-vanilla", roots["data"] / "dj-reachy",
                     roots["home"] / ".local" / "share" / "dj-reachy", roots["cache"] / "dj-reachy"]
    for root in vanilla_roots:
        (root / "songs").mkdir(parents=True)
        (root / "settings.json").write_text('{"music": {"duration_ms": 54000}}')
        (root / "songs" / "keep-song.bin").write_bytes(b"vanilla-song-do-not-touch")
        (root / "hf").mkdir()
        (root / "hf" / "keep-cache.txt").write_text("vanilla-cache-do-not-touch")
    before = {root: _snapshot(root) for root in vanilla_roots}
    env["DJ_REACHY_HOME"] = str(vanilla_roots[0])
    expected_home = roots["data"] / "dj-reachy-remix"
    expected_cache = roots["cache"] / "dj-reachy-remix" / "hf"
    if selection == "home":
        env.pop("XDG_DATA_HOME")
        env.pop("XDG_CACHE_HOME")
        expected_home = roots["home"] / ".local" / "share" / "dj-reachy-remix"
        expected_cache = roots["home"] / ".cache" / "dj-reachy-remix" / "hf"
    elif selection == "explicit":
        expected_home = tmp_path / "explicit-remix"
        env["DJ_REACHY_REMIX_HOME"] = str(expected_home)
    elif selection == "empty":
        env["DJ_REACHY_REMIX_HOME"] = ""
    elif selection == "expanded":
        env["DJ_REACHY_REMIX_HOME"] = "~/expanded-remix"
        expected_home = roots["home"] / "expanded-remix"
    elif selection == "hf_override":
        expected_cache = tmp_path / "explicit-hf"
        env["HF_HOME"] = str(expected_cache)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _CONFIG_PROBE, str(PROJECT_ROOT / "src"),
         str(tmp_path), str(expected_home), str(expected_cache)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    # Check the sentinels even when the child fails, so leakage is not hidden by
    # an earlier import/path assertion failure.
    assert {root: _snapshot(root) for root in vanilla_roots} == before
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "REMIX_CONFIG_ISOLATION_OK"
    assert json.loads((expected_home / "settings.json").read_text())["music"]["duration_ms"] == 10000
    assert (expected_home / "songs" / "fixture-song.bin").read_bytes() == b"offline-song-fixture"
    assert (expected_cache / "fixture-cache.txt").read_text() == "offline-cache-fixture"
