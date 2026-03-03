from .demucs_nodes import DemucsAudioSeparator, LoadAudioDirectory

NODE_CLASS_MAPPINGS = {
    "DemucsAudioSeparator": DemucsAudioSeparator,
    "LoadAudioDirectory": LoadAudioDirectory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DemucsAudioSeparator": "🎵 Demucs Audio Separator ⚡",
    "LoadAudioDirectory": "📂 Load Audio Directory ⚡"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
