from .demucs_nodes import DemucsProNode

NODE_CLASS_MAPPINGS = {
    "DemucsProNode": DemucsProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DemucsProNode": "Demucs Pro (Audio Separation)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
