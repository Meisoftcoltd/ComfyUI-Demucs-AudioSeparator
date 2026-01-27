import torch
import torchaudio
import os
import json
import numpy as np
from pathlib import Path

# Try to import folder_paths from ComfyUI
try:
    import folder_paths
except ImportError:
    # Fallback for development/testing outside ComfyUI
    class folder_paths:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        @staticmethod
        def get_output_directory():
            return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

from demucs.pretrained import get_model
from demucs.apply import apply_model

# Global cache for Demucs models to support fast swapping
_MODEL_CACHE = {}

class DemucsProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdxc", "mdxc_fb_ft"],),
                "device": (["cuda", "cpu"],),
                "shifts": ("INT", {"default": 1, "min": 1, "max": 10}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 0.9, "step": 0.05}),
                "split": ("BOOLEAN", {"default": True}),
                "vocals": ("BOOLEAN", {"default": True}),
                "drums": ("BOOLEAN", {"default": True}),
                "bass": ("BOOLEAN", {"default": True}),
                "other": ("BOOLEAN", {"default": True}),
                "guitar": ("BOOLEAN", {"default": False}),
                "piano": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "JSON")
    RETURN_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano", "metadata")
    FUNCTION = "separate"
    CATEGORY = "🎵 Audio/Separation"

    def separate(self, audio, model, device, shifts, overlap, split, vocals, drums, bass, other, guitar, piano):
        # Configure model path
        demucs_models_path = os.path.join(folder_paths.models_dir, "demucs")
        if not os.path.exists(demucs_models_path):
            os.makedirs(demucs_models_path, exist_ok=True)

        torch.hub.set_dir(demucs_models_path)

        # Determine device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚡ CUDA not available, falling back to CPU")
            device = "cpu"

        device_obj = torch.device(device)

        # Load model
        global _MODEL_CACHE
        if model in _MODEL_CACHE:
            model_inst = _MODEL_CACHE[model]
        else:
            print(f"⚡ Loading Demucs model: {model}...")
            try:
                model_inst = get_model(model)
            except Exception as e:
                raise RuntimeError(f"Failed to load Demucs model {model}: {str(e)}")

            # Optimizations for RTX 3090/Ampere
            if device == "cuda":
                model_inst.to(dtype=torch.bfloat16)

            # Limit cache size to 2 models to prevent OOM
            if len(_MODEL_CACHE) >= 2:
                oldest_model = next(iter(_MODEL_CACHE))
                del _MODEL_CACHE[oldest_model]

            _MODEL_CACHE[model] = model_inst

        model_inst.to(device_obj)
        model_inst.eval()

        # Prepare audio
        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        # Ensure waveform is [batch, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(device_obj)

        # Resample if necessary
        if sr != model_inst.samplerate:
            print(f"⚡ Resampling audio from {sr} to {model_inst.samplerate}...")
            resampler = torchaudio.transforms.Resample(sr, model_inst.samplerate).to(device_obj)
            waveform = resampler(waveform)

        # Apply model
        print(f"⚡ Separating audio with {model} (shifts={shifts}, overlap={overlap}, split={split})...")
        try:
            with torch.no_grad():
                # apply_model expects [batch, channels, samples] or [channels, samples]
                # It returns [batch, sources, channels, samples]
                out = apply_model(model_inst, waveform, shifts=shifts, split=split, overlap=overlap, progress=True, device=device_obj)
        except Exception as e:
            raise RuntimeError(f"Error during Demucs inference: {str(e)}")

        # out shape: [batch, sources, channels, samples]
        sources = model_inst.sources
        results = {}
        for i, source_name in enumerate(sources):
            results[source_name] = {
                "waveform": out[:, i, :, :].cpu(),
                "sample_rate": model_inst.samplerate
            }

        # Helper to get stem or zeroed audio
        def get_stem(name, enabled):
            if enabled and name in results:
                return results[name]
            else:
                # Return zeroed audio with same length and batch size
                batch_size = out.shape[0]
                channels = out.shape[2]
                samples = out.shape[3]
                return {
                    "waveform": torch.zeros((batch_size, channels, samples)),
                    "sample_rate": model_inst.samplerate
                }

        # Map to outputs
        out_vocals = get_stem("vocals", vocals)
        out_drums = get_stem("drums", drums)
        out_bass = get_stem("bass", bass)
        out_other = get_stem("other", other)
        out_guitar = get_stem("guitar", guitar)
        out_piano = get_stem("piano", piano)

        metadata = {
            "model": model,
            "device": str(device_obj),
            "shifts": shifts,
            "overlap": overlap,
            "split": split,
            "available_sources": sources,
            "input_sample_rate": sr,
            "output_sample_rate": model_inst.samplerate,
            "status": "success"
        }

        return (out_vocals, out_drums, out_bass, out_other, out_guitar, out_piano, metadata)
