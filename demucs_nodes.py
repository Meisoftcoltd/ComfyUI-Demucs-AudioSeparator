import torch
import torchaudio
import demucs.pretrained
import demucs.apply
import os
import logging

# ⚡ [Bolt] Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Bolt")

def bolt_log(msg):
    logger.info(f"⚡ [Bolt] {msg}")

# Global cache for models to support fast swapping
_MODEL_CACHE = {}

class DemucsSeparator:
    """
    🎧 Demucs Audio Separator - Optimized for RTX 3090 by Bolt ⚡
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi"], {"default": "htdemucs"}),
                "shifts": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
                "split": ("BOOLEAN", {"default": True}),
                "segment": ("INT", {"default": 7, "min": 1, "max": 60, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "JSON")
    RETURN_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano", "debug_info")
    FUNCTION = "separate"
    CATEGORY = "🎵 Audio/Separation"

    def __init__(self):
        self.device = self._get_optimal_device()
        self.dtype = self._get_optimal_dtype()
        bolt_log(f"Initialized on {self.device} with {self.dtype} 🚀")

    def _get_optimal_device(self):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "RTX 3090" in device_name:
                bolt_log("Detected RTX 3090 - Unleashing maximum performance! ⚡")
            return "cuda"
        return "cpu"

    def _get_optimal_dtype(self):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            # Ampere (30 series) and newer support bfloat16
            if any(x in device_name for x in ["RTX 30", "RTX 40", "A100", "A10", "A30"]):
                bolt_log("Ampere/Ada architecture detected, using bfloat16 for speed 💎")
                return torch.bfloat16
        return torch.float32

    def separate(self, audio, model_name, shifts, overlap, split, segment):
        bolt_log(f"Starting separation with model: {model_name} 🎧")

        waveform = audio["waveform"] # [B, C, S]
        sample_rate = audio["sample_rate"]

        # Load Model
        if model_name in _MODEL_CACHE:
            bolt_log(f"Model {model_name} found in cache. Swapping... 🔄")
            model = _MODEL_CACHE[model_name]
        else:
            bolt_log(f"Loading model {model_name}... 📂")
            model = demucs.pretrained.get_model(model_name)
            # Pin memory for parameters to speed up transfers from RAM to VRAM
            # This takes advantage of the 128GB of RAM
            for p in model.parameters():
                p.data = p.data.pin_memory()
            bolt_log("Model parameters pinned to shared memory for ultra-fast VRAM loading 🚀")
            _MODEL_CACHE[model_name] = model

        model.to(self.device)

        # Resampling if needed
        if sample_rate != model.samplerate:
            bolt_log(f"Resampling from {sample_rate}Hz to {model.samplerate}Hz... 🔄")
            resampler = torchaudio.transforms.Resample(sample_rate, model.samplerate).to(self.device)
            waveform = resampler(waveform.to(self.device))
            sample_rate = model.samplerate

        # Apply bfloat16 if optimized
        if self.dtype == torch.bfloat16:
            model.to(dtype=torch.bfloat16)
            waveform = waveform.to(dtype=torch.bfloat16)
        else:
            waveform = waveform.to(self.device)

        bolt_log(f"Processing {waveform.shape[0]} audio batch(es). VRAM mode active. ⚡")

        # Perform separation
        try:
            # apply_model returns [batch, sources, channels, samples]
            separated_batch = demucs.apply.apply_model(
                model,
                waveform,
                shifts=shifts,
                split=split,
                overlap=overlap,
                segment=segment,
                device=self.device,
                progress=True
            )
        except Exception as e:
            bolt_log(f"Separation failed: {str(e)} ❌")
            raise e

        bolt_log("Separation complete! Formatting outputs... ✅")

        # Prepare outputs
        def get_stem_audio(stem_name):
            if stem_name not in model.sources:
                # Return silent audio if stem is missing
                silent = torch.zeros((waveform.shape[0], 2, waveform.shape[-1]), dtype=torch.float32)
                return {"waveform": silent, "sample_rate": model.samplerate}

            idx = model.sources.index(stem_name)
            tensor = separated_batch[:, idx] # [B, C, S]
            # Return to float32 for ComfyUI compatibility
            tensor = tensor.to(torch.float32).cpu()
            return {"waveform": tensor, "sample_rate": model.samplerate}

        out_vocals = get_stem_audio("vocals")
        out_drums = get_stem_audio("drums")
        out_bass = get_stem_audio("bass")
        out_other = get_stem_audio("other")
        out_guitar = get_stem_audio("guitar")
        out_piano = get_stem_audio("piano")

        debug_info = {
            "model": model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "stems_found": model.sources,
            "batch_size": waveform.shape[0],
            "bolt_optimized": True
        }

        return (out_vocals, out_drums, out_bass, out_other, out_guitar, out_piano, debug_info)

NODE_CLASS_MAPPINGS = {
    "DemucsSeparator": DemucsSeparator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DemucsSeparator": "🎧 Demucs Audio Separator"
}
