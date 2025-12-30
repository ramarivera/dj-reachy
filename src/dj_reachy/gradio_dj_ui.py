"""Gradio DJ UI components - Song library and Dance settings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gradio as gr

from .config import config


logger = logging.getLogger(__name__)


class DJReachyUI:
    """Container for DJ Reachy Gradio components - Songs and Dance settings."""

    def __init__(self) -> None:
        """Initialize the DJReachyUI instance."""
        self._songs_dir = config.SONGS_DIR

        # Components (initialized in create_components)
        self.songs_list: gr.Dataframe
        self.refresh_btn: gr.Button
        self.play_btn: gr.Button
        self.delete_btn: gr.Button
        self.song_audio: gr.Audio
        self.dance_intensity: gr.Slider
        self.status_md: gr.Markdown

    def _get_songs_metadata(self) -> dict:
        """Load songs metadata from JSON file."""
        meta_path = self._songs_dir / "metadata.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _list_songs(self) -> list[list[str]]:
        """Get list of songs with metadata."""
        songs = []
        metadata = self._get_songs_metadata()
        
        if self._songs_dir.exists():
            for f in sorted(self._songs_dir.glob("*.mp3"), key=lambda x: x.stat().st_mtime, reverse=True):
                stat = f.stat()
                name = f.stem
                parts = name.split("_", 1)
                timestamp = parts[0] if parts[0].isdigit() else ""
                default_title = parts[1].replace("_", " ") if len(parts) > 1 else name
                
                song_meta = metadata.get(f.name, {})
                title = song_meta.get("title", default_title)
                
                # Format size
                size_bytes = stat.st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                
                songs.append([f.name, title, size_str])
        
        return songs if songs else [["", "No songs yet. Ask DJ Reachy to make you a song!", ""]]

    def _get_song_path(self, filename: str) -> str | None:
        """Get full path to a song file."""
        if not filename:
            return None
        song_path = self._songs_dir / filename
        if song_path.exists() and song_path.suffix == ".mp3":
            return str(song_path)
        return None

    def _delete_song(self, filename: str) -> tuple[list[list[str]], str]:
        """Delete a song file."""
        if not filename:
            return self._list_songs(), "No song selected"
        
        song_path = self._songs_dir / filename
        if song_path.exists() and song_path.suffix == ".mp3":
            try:
                song_path.unlink()
                # Also remove from metadata
                metadata = self._get_songs_metadata()
                if filename in metadata:
                    del metadata[filename]
                    meta_path = self._songs_dir / "metadata.json"
                    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                logger.info("Deleted song: %s", filename)
                return self._list_songs(), f"✅ Deleted: {filename}"
            except Exception as e:
                logger.warning("Failed to delete song %s: %s", filename, e)
                return self._list_songs(), f"❌ Failed to delete: {e}"
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
        """Set dance intensity from slider value (0-100)."""
        try:
            from dj_reachy.profiles.dj_reachy.make_song_and_dance import set_dance_intensity
            # Convert slider (0-100) to internal (0-0.25)
            internal = slider_value / 400
            set_dance_intensity(internal)
            logger.info("Set dance intensity to %.2f (slider: %d%%)", internal, int(slider_value))
            return f"✅ Dance intensity: {int(slider_value)}%"
        except Exception as e:
            logger.warning("Failed to set dance intensity: %s", e)
            return f"❌ Failed: {e}"

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
                return path, f"🎵 Playing: {title}"
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

            # Load initial song list on page load
            blocks.load(
                fn=_refresh_songs,
                inputs=[],
                outputs=[self.songs_list],
            )

