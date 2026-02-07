import torch
import torchaudio
import os
import json
import numpy as np
import inspect
import tempfile
import gc
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
        @staticmethod
        def get_temp_directory():
             return tempfile.gettempdir()

if not hasattr(folder_paths, 'get_temp_directory'):
    folder_paths.get_temp_directory = lambda: tempfile.gettempdir()

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
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 0.99, "step": 0.05}),
                "split": ("BOOLEAN", {"default": True}),
                "vocals": ("BOOLEAN", {"default": True}),
                "drums": ("BOOLEAN", {"default": True}),
                "bass": ("BOOLEAN", {"default": True}),
                "other": ("BOOLEAN", {"default": True}),
                "guitar": ("BOOLEAN", {"default": False}),
                "piano": ("BOOLEAN", {"default": False}),
                "segment_strategy": (["Auto", "Disk", "RAM"], {"default": "Auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "JSON")
    RETURN_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano", "metadata")
    FUNCTION = "separate"
    CATEGORY = "🎵 Audio/Separation"

    def separate(self, audio, model, device, precision, shifts, overlap, split, vocals, drums, bass, other, guitar, piano, segment_strategy="Auto"):
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

        # Check strategy
        use_disk = False
        duration_seconds = waveform.shape[-1] / sr

        if segment_strategy == "Disk":
            use_disk = True
        elif segment_strategy == "Auto":
            # If > 3 minutes (conservative), use disk to be safe on lower RAM systems,
            # or if the user specifically has OOM issues with large files.
            if duration_seconds > 180:
                use_disk = True

        if use_disk:
             print(f"⚡ [Demucs Pro] Using Disk-Based Processing for {duration_seconds:.2f}s audio.")
             return self._process_segmented(
                 model_inst, waveform, sr, device_obj, dtype, shifts, overlap, split,
                 vocals, drums, bass, other, guitar, piano, model_name, precision
             )

        # --- RAM Strategy (Legacy) ---

        # Move waveform to device and precision
        # Ensure that if we are resampling later, we might need float32 input for kernels.
        # But if we move to device, we cast to 'dtype'.

        # If input is float16/Half (e.g. from previous nodes), and we want to process it.

        # NOTE: torchaudio.transforms.Resample usually prefers Float32.
        # If dtype is float16, we might get the mismatch error if we resample.
        # So we should convert to float32 IF resampling is needed, then convert back to desired dtype?

        needs_resample = (sr != model_inst.samplerate)

        if needs_resample:
             # Ensure float32 for high quality resampling and kernel compatibility
             waveform = waveform.to(device_obj).to(torch.float32)
             print(f"⚡ Resampling audio from {sr} to {model_inst.samplerate}...")
             resampler = torchaudio.transforms.Resample(sr, model_inst.samplerate).to(device_obj)
             waveform = resampler(waveform)
             # Now convert to target dtype for inference if needed
             if dtype != torch.float32:
                 waveform = waveform.to(dtype)
        else:
             # No resample, just move and cast
             waveform = waveform.to(device_obj).to(dtype)

        # Progress bar integration
        pbar = None
        if HAS_COMFY:
            pbar = comfy.utils.ProgressBar(100)

        def progress_callback(info):
            if pbar:
                # Support various keys from different Demucs versions
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

        return self._format_results(out, model_inst, model_name, device_obj, precision, shifts, overlap, split, sr, vocals, drums, bass, other, guitar, piano)

    def _process_segmented(self, model_inst, waveform, sr, device_obj, dtype, shifts, overlap, split, vocals, drums, bass, other, guitar, piano, model_name, precision):
        # Prepare output storage on disk
        temp_dir = folder_paths.get_temp_directory()
        import uuid
        job_dir = os.path.join(temp_dir, f"demucs_{uuid.uuid4().hex}")
        os.makedirs(job_dir, exist_ok=True)
        print(f"⚡ [Demucs Pro] Created temp cache at {job_dir}")

        batch, channels, input_length = waveform.shape
        target_sr = model_inst.samplerate
        output_length = int(input_length * target_sr / sr) + 1

        sources = model_inst.sources

        memmaps = {}
        for source in sources:
             path = os.path.join(job_dir, f"{source}.npy")
             memmaps[source] = np.memmap(path, dtype='float32', mode='w+', shape=(batch, channels, output_length))

        chunk_duration = 120
        margin_duration = 10

        chunk_samples = int(chunk_duration * sr)
        margin_samples = int(margin_duration * sr)

        total_samples = input_length
        num_chunks = int(np.ceil(total_samples / chunk_samples))

        pbar = None
        if HAS_COMFY:
            pbar = comfy.utils.ProgressBar(total_samples)

        resampler = None
        if sr != target_sr:
             # Ensure resampler is float32
             resampler = torchaudio.transforms.Resample(sr, target_sr).to(device_obj).to(torch.float32)

        current_sample = 0
        output_sample_offset = 0

        for i in range(num_chunks):
            start = current_sample
            end = min(current_sample + chunk_samples, total_samples)

            pad_left = 0
            pad_right = 0

            s_start = start - margin_samples
            if s_start < 0:
                pad_left = -s_start
                s_start = 0

            s_end = end + margin_samples
            if s_end > total_samples:
                pad_right = s_end - total_samples
                s_end = total_samples

            # Slice input (CPU)
            chunk_wave = waveform[:, :, s_start:s_end]

            # Move to device
            # Ensure we are float32 BEFORE resampling if needed
            chunk_wave = chunk_wave.to(device_obj).to(torch.float32)

            if pad_left > 0 or pad_right > 0:
                 chunk_wave = torch.nn.functional.pad(chunk_wave, (pad_left, pad_right))

            if resampler:
                chunk_wave = resampler(chunk_wave)

            # Now cast to target dtype if different (e.g. float16)
            # though apply_model usually wants float32 anyway (we force it inside).
            if dtype != torch.float32:
                chunk_wave = chunk_wave.to(dtype)

            try:
                with torch.no_grad():
                    chunk_out = apply_model(model_inst, chunk_wave.to(torch.float32),
                                            shifts=shifts, split=split, overlap=overlap,
                                            progress=False, device=device_obj)
            except Exception as e:
                 print(f"Error processing chunk {i}: {e}")
                 raise e

            ratio = target_sr / sr

            offset_in_output = int(margin_samples * ratio)
            length_in_output = int((end - start) * ratio)

            extracted = chunk_out[:, :, :, offset_in_output : offset_in_output + length_in_output]

            extracted_np = extracted.cpu().numpy()

            write_len = extracted_np.shape[-1]
            if output_sample_offset + write_len > memmaps[sources[0]].shape[-1]:
                 write_len = memmaps[sources[0]].shape[-1] - output_sample_offset
                 extracted_np = extracted_np[..., :write_len]

            for s_idx, source in enumerate(sources):
                 memmaps[source][:, :, output_sample_offset : output_sample_offset + write_len] = extracted_np[:, s_idx, :, :]
                 memmaps[source].flush()

            output_sample_offset += write_len
            current_sample = end

            if pbar:
                pbar.update_absolute(current_sample)

            del chunk_wave, chunk_out, extracted
            torch.cuda.empty_cache()
            gc.collect()

        results = {}
        for source in sources:
            mm = memmaps[source]
            mm_trimmed = mm[:, :, :output_sample_offset]
            tensor = torch.from_numpy(mm_trimmed)
            results[source] = {
                "waveform": tensor,
                "sample_rate": target_sr
            }

        final_batch_size = batch
        final_channels = channels
        final_samples = output_sample_offset

        def get_stem_segmented(name, enabled):
            if enabled and name in results:
                return results[name]
            else:
                zero_path = os.path.join(job_dir, "zeros.npy")
                if not os.path.exists(zero_path):
                     z = np.memmap(zero_path, dtype='float32', mode='w+', shape=(final_batch_size, final_channels, final_samples))
                     z.flush()
                     del z
                z_read = np.memmap(zero_path, dtype='float32', mode='r', shape=(final_batch_size, final_channels, final_samples))
                return {
                    "waveform": torch.from_numpy(z_read),
                    "sample_rate": target_sr
                }

        out_vocals = get_stem_segmented("vocals", vocals)
        out_drums = get_stem_segmented("drums", drums)
        out_bass = get_stem_segmented("bass", bass)
        out_other = get_stem_segmented("other", other)
        out_guitar = get_stem_segmented("guitar", guitar)
        out_piano = get_stem_segmented("piano", piano)

        metadata = {
            "model": model_name,
            "device": str(device_obj),
            "precision": precision,
            "shifts": shifts,
            "overlap": overlap,
            "split": split,
            "available_sources": sources,
            "input_sample_rate": sr,
            "output_sample_rate": target_sr,
            "status": "success",
            "strategy": "disk_segmented",
            "temp_dir": job_dir
        }

        print(f"⚡ [Demucs Pro] Disk-based separation completed.")
        return (out_vocals, out_drums, out_bass, out_other, out_guitar, out_piano, metadata)

    def _format_results(self, out, model_inst, model_name, device_obj, precision, shifts, overlap, split, sr, vocals, drums, bass, other, guitar, piano):
        sources = model_inst.sources
        results = {}
        for i, source_name in enumerate(sources):
            results[source_name] = {
                "waveform": out[:, i, :, :].to(torch.float32).cpu(),
                "sample_rate": model_inst.samplerate
            }

        def get_stem(name, enabled):
            if enabled and name in results:
                return results[name]
            else:
                batch_size = out.shape[0]
                channels = out.shape[2]
                samples = out.shape[3]
                return {
                    "waveform": torch.zeros((batch_size, channels, samples), dtype=torch.float32),
                    "sample_rate": model_inst.samplerate
                }

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
