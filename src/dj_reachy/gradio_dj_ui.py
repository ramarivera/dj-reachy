"""Gradio DJ UI components - Song library and Dance settings."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import gradio as gr

from .config import config


logger = logging.getLogger(__name__)


class DJReachyUI:
    """Container for DJ Reachy Gradio components - Songs and Dance settings."""

    def __init__(self) -> None:
        """Initialize the DJReachyUI instance."""
        self._songs_dir = config.SONGS_DIR
        self._selected_filename: str | None = None

        # Components (initialized in create_components)
        self.songs_list: gr.Dataframe
        self.refresh_btn: gr.Button
        self.play_btn: gr.Button
        self.delete_btn: gr.Button
        self.song_audio: gr.Audio
        self.dance_intensity: gr.Slider
        self.status_md: gr.Markdown
        self.play_btn: gr.Button
        self.pause_btn: gr.Button
        self.resume_btn: gr.Button
        self.stop_btn: gr.Button

    def _get_songs_metadata(self) -> dict:
        """Load metadata only from a regular file inside the owned library."""
        meta_path = self._songs_dir / "metadata.json"
        try:
            if meta_path.is_symlink() or meta_path.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                return {}
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, RuntimeError, json.JSONDecodeError):
            return {}

    def _list_songs(self) -> list[list[str]]:
        """Get regular MP3 tracks with their saved metadata."""
        songs = []
        metadata = self._get_songs_metadata()
        if self._songs_dir.exists():
            for song_path in sorted(self._songs_dir.glob("*.mp3"), key=lambda item: item.name, reverse=True):
                try:
                    if song_path.is_symlink() or song_path.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                        continue
                    stat = song_path.stat()
                except (OSError, RuntimeError):
                    continue
                name = song_path.stem
                parts = name.split("_", 1)
                default_title = parts[1].replace("_", " ") if len(parts) > 1 else name
                song_meta = metadata.get(song_path.name, {})
                title = song_meta.get("title", default_title)
                size_bytes = stat.st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                songs.append([song_path.name, title, size_str])
        return songs if songs else [["", "No songs yet. Ask DJ Reachy to make you a song!", ""]]

    def _get_song_path(self, filename: str) -> str | None:
        """Resolve only regular MP3 tracks directly inside the library."""
        if not filename or Path(filename).name != filename:
            return None
        song_path = self._songs_dir / filename
        if song_path.is_symlink() or song_path.suffix.lower() != ".mp3" or not song_path.is_file():
            return None
        try:
            if song_path.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                return None
        except (OSError, RuntimeError):
            return None
        return str(song_path)

    def _delete_song(self, filename: str) -> tuple[list[list[str]], str]:
        """Delete a song file."""
        if not filename:
            return self._list_songs(), "No song selected"
        
        song_path_value = self._get_song_path(filename)
        if song_path_value:
            song_path = Path(song_path_value)
            metadata_path = self._songs_dir / "metadata.json"
            try:
                if metadata_path.is_symlink():
                    return self._list_songs(), "Metadata file is a symlink; track was not deleted."
                if metadata_path.exists() and metadata_path.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                    return self._list_songs(), "Metadata file is outside the library; track was not deleted."
                sidecar = song_path.with_suffix(".json")
                if sidecar.exists() and sidecar.resolve(strict=True).parent != self._songs_dir.resolve(strict=True):
                    return self._list_songs(), "Track metadata is outside the library; track was not deleted."
                song_path.unlink()
                sidecar.unlink(missing_ok=True)
                metadata = self._get_songs_metadata()
                if filename in metadata:
                    del metadata[filename]
                    fd, temp_name = tempfile.mkstemp(prefix=".metadata.", suffix=".tmp", dir=self._songs_dir)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as stream:
                            json.dump(metadata, stream, indent=2)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temp_name, metadata_path)
                    finally:
                        Path(temp_name).unlink(missing_ok=True)
                logger.info("Deleted song: %s", filename)
                return self._list_songs(), f"Deleted: {filename}"
            except (OSError, RuntimeError) as exc:
                logger.warning("Failed to delete song %s: %s", filename, type(exc).__name__)
                return self._list_songs(), "Failed to delete the saved track."
        return self._list_songs(), "❌ Song not found"

    def _get_dance_intensity(self) -> float:
        """Get current dance intensity (0-100 for slider)."""
        try:
            from dj_reachy.profiles.dj_reachy.make_song_and_dance import get_dance_intensity
            # Internal: 0-0.25, Slider: 0-100
            internal = get_dance_intensity()
            return internal * 400  # Convert to slider scale
        except Exception:
            return 80  # Default 80%

    def _set_dance_intensity(self, slider_value: float) -> str:
        """Persist bounded motion intensity and apply it to the active dancer."""
        try:
            from dj_reachy.config import update_runtime_settings
            from dj_reachy.profiles.dj_reachy.make_song_and_dance import set_dance_intensity
            intensity = max(0.0, min(100.0, float(slider_value))) / 400.0
            update_runtime_settings({"motion": {"intensity": intensity}})
            set_dance_intensity(intensity)
            return f"Dance intensity: {int(slider_value)}%"
        except Exception as exc:
            logger.warning("Failed to save dance intensity: %s", type(exc).__name__)
            return "Failed to save dance intensity."

    def create_components(self) -> None:
        """Instantiate Gradio components for the DJ UI."""
        # Status message
        self.status_md = gr.Markdown(value="", visible=True)
        
        # Dance intensity slider
        self.dance_intensity = gr.Slider(
            label="🎚️ Dance Intensity",
            minimum=0,
            maximum=100,
            value=self._get_dance_intensity(),
            step=1,
            info="Control how dramatically DJ Reachy dances to the music",
        )
        
        # Songs list
        self.songs_list = gr.Dataframe(
            headers=["Filename", "Title", "Size"],
            datatype=["str", "str", "str"],
            value=self._list_songs(),
            label="🎵 Song Library",
            interactive=False,
            wrap=True,
        )
        
        # Song controls
        self.refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
        self.play_btn = gr.Button("▶️ Play Selected", variant="primary", size="sm")
        self.delete_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm")
        
        # Audio player (hidden, used for playback)
        self.song_audio = gr.Audio(
            label="Now Playing",
            type="filepath",
            interactive=False,
            visible=True,
            autoplay=True,
        )

    def additional_inputs_ordered(self) -> list[Any]:
        """Return components needed as additional inputs (none for DJ UI)."""
        # We don't need to pass these to the Stream handler
        return []

    def wire_events(self, handler: Any, blocks: gr.Blocks) -> None:
        """Attach event handlers to components within a Blocks context."""

        def _refresh_songs() -> list[list[str]]:
            return self._list_songs()

        def _play_song(songs_data, evt: gr.SelectData) -> tuple[str | None, str]:
            if evt.index is None:
                return None, "Select a song to play"
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            
            # Handle both DataFrame and list formats
            try:
                if hasattr(songs_data, 'iloc'):
                    # It's a pandas DataFrame
                    if row_idx < 0 or row_idx >= len(songs_data):
                        return None, "Invalid selection"
                    filename = songs_data.iloc[row_idx, 0]  # First column
                    title = songs_data.iloc[row_idx, 1]     # Second column
                else:
                    # It's a list of lists
                    if row_idx < 0 or row_idx >= len(songs_data):
                        return None, "Invalid selection"
                    filename = songs_data[row_idx][0]
                    title = songs_data[row_idx][1]
            except (IndexError, KeyError) as e:
                logger.warning("Error accessing song data: %s", e)
                return None, "❌ Error accessing song data"
            
            if not filename:
                return None, "No valid song selected"
            path = self._get_song_path(filename)
            if path:
                self._selected_filename = filename
                return path, f"🎵 Selected: {title}. Press Play to send it to Reachy."
            return None, "❌ Song file not found"

        def _delete_selected(songs_data, evt: gr.SelectData | None) -> tuple[list[list[str]], str]:
            if evt is None or evt.index is None:
                return self._list_songs(), "Select a song to delete"
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            
            try:
                if hasattr(songs_data, 'iloc'):
                    # It's a pandas DataFrame
                    if row_idx < 0 or row_idx >= len(songs_data):
                        return self._list_songs(), "Invalid selection"
                    filename = songs_data.iloc[row_idx, 0]
                else:
                    # It's a list of lists
                    if row_idx < 0 or row_idx >= len(songs_data):
                        return self._list_songs(), "Invalid selection"
                    filename = songs_data[row_idx][0]
            except (IndexError, KeyError) as e:
                logger.warning("Error accessing song data for delete: %s", e)
                return self._list_songs(), "❌ Error accessing song data"
            
            return self._delete_song(filename)

        def _update_intensity(value: float) -> str:
            return self._set_dance_intensity(value)

        async def _play_selected() -> str:
            if not self._selected_filename:
                return "Select a saved track first."
            from dj_reachy.tools.core_tools import dispatch_tool_call
            result = await dispatch_tool_call(
                "play_saved_song", json.dumps({"filename": self._selected_filename}), handler.deps
            )
            return str(result.get("status") or result.get("error") or "Playback finished.")

        async def _music_control(tool_name: str) -> str:
            from dj_reachy.tools.core_tools import dispatch_tool_call
            result = await dispatch_tool_call(tool_name, "{}", handler.deps)
            return str(result.get("status") or result.get("message") or result.get("error") or "Done.")

        def _delete_current() -> tuple[list[list[str]], str]:
            if not self._selected_filename:
                return self._list_songs(), "Select a saved track first."
            rows, status = self._delete_song(self._selected_filename)
            self._selected_filename = None
            return rows, status

        with blocks:
            # Refresh button
            self.refresh_btn.click(
                fn=_refresh_songs,
                inputs=[],
                outputs=[self.songs_list],
            )
            
            # Play on row selection
            self.songs_list.select(
                fn=_play_song,
                inputs=[self.songs_list],
                outputs=[self.song_audio, self.status_md],
            )
            
            # Dance intensity slider
            self.dance_intensity.release(
                fn=_update_intensity,
                inputs=[self.dance_intensity],
                outputs=[self.status_md],
            )
            self.play_btn.click(fn=_play_selected, inputs=[], outputs=[self.status_md])
            self.pause_btn.click(fn=lambda: _music_control("pause_music"), inputs=[], outputs=[self.status_md])
            self.resume_btn.click(fn=lambda: _music_control("resume_music"), inputs=[], outputs=[self.status_md])
            self.stop_btn.click(fn=lambda: _music_control("stop_music"), inputs=[], outputs=[self.status_md])
            self.delete_btn.click(fn=_delete_current, inputs=[], outputs=[self.songs_list, self.status_md])

            # Load initial song list on page load
            blocks.load(
                fn=_refresh_songs,
                inputs=[],
                outputs=[self.songs_list],
            )

