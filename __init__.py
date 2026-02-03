from .demucs_nodes import DemucsAudioSeparator

NODE_CLASS_MAPPINGS = {
    "DemucsAudioSeparator": DemucsAudioSeparator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DemucsAudioSeparator": "🎵 Demucs Audio Separator ⚡"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
