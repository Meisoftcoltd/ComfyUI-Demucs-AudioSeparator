from .demucs_nodes import DemucsAudioSeparator

NODE_CLASS_MAPPINGS = {
    "DemucsAudioSeparator": DemucsAudioSeparator,
    "DemucsProNode": DemucsAudioSeparator  # Legacy support
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DemucsAudioSeparator": "🎵 Demucs Audio Separator ⚡",
    "DemucsProNode": "🎵 Demucs Audio Separator (Legacy) ⚡"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
