import torch
import torchaudio
import os
import json
import numpy as np
import inspect
from pathlib import Path

# Try to import ComfyUI utilities
try:
    import folder_paths
except ImportError:
    # Fallback for development/testing outside ComfyUI
    class folder_paths:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        @staticmethod
        def get_output_directory():
            return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

try:
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

from demucs.pretrained import get_model
from demucs.apply import apply_model

class DemucsAudioSeparator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "mdxc", "mdxc_fb_ft"],),
                "device": (["cuda", "cpu"],),
                "precision": (["float32", "float16"], {"default": "float32"}),
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

    def separate(self, audio, model, device, precision, shifts, overlap, split, vocals, drums, bass, other, guitar, piano):
        model_name = model

        # Legacy support and safety check for bfloat16
        if precision == "bfloat16":
            print("⚠️ [Demucs Pro] BFloat16 not supported for FFT. Upgrading to Float32 for inference.")
            precision = "float32"

        if precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

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
        print(f"⚡ Loading Demucs model: {model}...")
        try:
            model_inst = get_model(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load Demucs model {model}: {str(e)}")

        model_inst.to(device_obj)
        model_inst.eval()

        # Prepare audio
        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        # Ensure waveform is [batch, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Move waveform to device and precision
        waveform = waveform.to(device_obj).to(dtype)

        # Resample if necessary
        if sr != model_inst.samplerate:
            print(f"⚡ Resampling audio from {sr} to {model_inst.samplerate}...")
            resampler = torchaudio.transforms.Resample(sr, model_inst.samplerate).to(device_obj)
            waveform = resampler(waveform)

        # Progress bar integration
        pbar = None
        if HAS_COMFY:
            pbar = comfy.utils.ProgressBar(100)

        def progress_callback(info):
            if pbar:
                # Support various keys from different Demucs versions
                # e.g., 'progress', 'shift', 'total', 'shift_idx'
                total = info.get('total') or info.get('shifts') or shifts
                current = info.get('shift') or info.get('shift_idx') or info.get('progress')

                if isinstance(current, float) and current <= 1.0:
                    pbar.update_absolute(int(current * 100))
                elif total and current is not None:
                    # current might be 0-indexed
                    p_val = min(100, int((current + 1) / total * 100))
                    pbar.update_absolute(p_val)

        # Apply model
        print(f"⚡ Separating audio with {model} (shifts={shifts}, overlap={overlap}, split={split})...")
        try:
            with torch.no_grad():
                # Check if the installed version of apply_model supports a callback
                apply_kwargs = {
                    "shifts": shifts,
                    "split": split,
                    "overlap": overlap,
                    "progress": False,
                    "device": device_obj,
                }

                sig = inspect.signature(apply_model)
                if 'callback' in sig.parameters:
                    apply_kwargs["callback"] = progress_callback

                # Force float32 for inference to avoid cuFFT/BFloat16 issues
                out = apply_model(model_inst, waveform.to(torch.float32), **apply_kwargs)
        except Exception as e:
            raise RuntimeError(f"⚡ [Demucs Pro] Error during Demucs inference: {str(e)}")

        # out shape: [batch, sources, channels, samples]
        sources = model_inst.sources
        results = {}
        for i, source_name in enumerate(sources):
            # Convert back to float32 for output and move to CPU
            results[source_name] = {
                "waveform": out[:, i, :, :].to(torch.float32).cpu(),
                "sample_rate": model_inst.samplerate
            }

        # Helper to get stem or zeroed audio based on user selection
        def get_stem(name, enabled):
            if enabled and name in results:
                return results[name]
            else:
                # Return zeroed audio with same length and batch size if stem is disabled or unavailable
                batch_size = out.shape[0]
                channels = out.shape[2]
                samples = out.shape[3]
                return {
                    "waveform": torch.zeros((batch_size, channels, samples), dtype=torch.float32),
                    "sample_rate": model_inst.samplerate
                }

        # Map results to discrete outputs, respecting the boolean flags
        out_vocals = get_stem("vocals", vocals)
        out_drums = get_stem("drums", drums)
        out_bass = get_stem("bass", bass)
        out_other = get_stem("other", other)
        out_guitar = get_stem("guitar", guitar)
        out_piano = get_stem("piano", piano)

        metadata = {
            "model": model_name,
            "device": str(device_obj),
            "precision": precision,
            "shifts": shifts,
            "overlap": overlap,
            "split": split,
            "available_sources": sources,
            "input_sample_rate": sr,
            "output_sample_rate": model_inst.samplerate,
            "status": "success"
        }

        print(f"⚡ [Demucs Pro] Separation completed successfully.")
        return (out_vocals, out_drums, out_bass, out_other, out_guitar, out_piano, metadata)
