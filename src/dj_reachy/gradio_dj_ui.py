"""Saved-track presentation shared by the managed and browser control panels."""

import json
from typing import Any
from pathlib import Path

from dj_reachy.config import config
from dj_reachy.runtime_settings import atomic_write_private


class DJReachyUI:
    """List and delete regular library files without following symbolic links."""

    def __init__(self) -> None:
        """Use the configured user-owned library, never the package directory."""
        self._songs_dir = config.SONGS_DIR

    def _read_metadata(self, path: Path) -> dict[str, Any]:
        try:
            if path.is_symlink() or path.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                return {}
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, RuntimeError):
            return {}

    def _get_songs_metadata(self) -> dict[str, Any]:
        return self._read_metadata(self._songs_dir / "metadata.json")

    def _list_songs(self) -> list[list[str]]:
        if self._songs_dir.is_symlink() or not self._songs_dir.is_dir():
            return []
        metadata = self._get_songs_metadata()
        songs: list[list[str]] = []
        for song in sorted(self._songs_dir.glob("*.mp3"), key=lambda path: path.name, reverse=True):
            path = self._get_song_path(song.name)
            if path is None:
                continue
            try:
                size = song.stat().st_size
            except OSError:
                continue
            default_title = song.stem.split("_", 1)[-1].replace("_", " ")
            legacy = metadata.get(song.name, {})
            legacy = legacy if isinstance(legacy, dict) else {}
            sidecar = self._read_metadata(song.with_suffix(".json"))
            title = legacy.get("title", sidecar.get("title", default_title))
            if not isinstance(title, str):
                title = default_title
            size_label = f"{size} B" if size < 1024 else (
                f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            )
            songs.append([song.name, title[:1000], size_label])
        return songs

    def _get_song_path(self, filename: str) -> str | None:
        if not filename or Path(filename).name != filename or self._songs_dir.is_symlink():
            return None
        song = self._songs_dir / filename
        try:
            if song.is_symlink() or song.suffix.lower() != ".mp3" or not song.is_file():
                return None
            if song.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                return None
        except (OSError, ValueError, RuntimeError):
            return None
        return str(song)

    def _delete_song(self, filename: str) -> tuple[list[list[str]], str]:
        path = self._get_song_path(filename)
        if path is None:
            return self._list_songs(), "Saved track not found."
        song = Path(path)
        metadata_path, sidecar = self._songs_dir / "metadata.json", song.with_suffix(".json")
        try:
            for metadata_file in (metadata_path, sidecar):
                if metadata_file.is_symlink() or (
                    metadata_file.exists() and metadata_file.resolve(strict=True).parent != self._songs_dir.resolve(strict=True)
                ):
                    return self._list_songs(), "Track metadata is a symlink or outside the library; nothing deleted."
            metadata = self._get_songs_metadata()
            if filename in metadata:
                del metadata[filename]
                atomic_write_private(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            sidecar.unlink(missing_ok=True)
            song.unlink()
            return self._list_songs(), "Saved track deleted."
        except (OSError, ValueError, RuntimeError):
            return self._list_songs(), "Could not delete the saved track."
