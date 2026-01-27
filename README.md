# Bolt ⚡ - ComfyUI Demucs Pro Edition

A high-performance implementation of Meta Demucs v4 for ComfyUI, optimized for extreme speed on high-end hardware like the **NVIDIA RTX 3090**.

## 🚀 Features

- **🎧 Stem Separation**: Split audio into Vocals, Drums, Bass, Other, and even Guitar/Piano (with 6s models).
- **⚡ Bolt Optimized**: Measurably faster on RTX 30/40 series GPUs using `bfloat16` and VRAM optimizations.
- **🔄 Fast Swapping**: Persistent model caching and `pin_memory` for instantaneous switching between models.
- **🎨 Polished UI**: Emoji-rich node names and categories for a clean, scannable workflow.
- **🎙️ Long Audio Support**: Automatic split strategy with overlap to handle long tracks without memory crashes.

## 📦 Installation

1. Navigate to your ComfyUI `custom_nodes` folder.
2. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/ComfyUI-Demucs-Pro.git
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Nodes

### 🎧 Demucs Audio Separator
- **Category**: `🎵 Audio/Separation`
- **Inputs**:
  - `audio`: The input audio signal.
  - `model_name`: Choose between `htdemucs` (standard), `htdemucs_ft` (high quality), `htdemucs_6s` (6 stems), and `hdemucs_mmi`.
  - `shifts`: Quality multiplier (higher is better but slower).
  - `overlap`: Overlap between segments.
  - `split`: Enable/disable split strategy.
  - `segment`: Segment length in seconds.

## 📊 Bolt Optimization Status

| Hardware | Status | Optimization |
| :--- | :--- | :--- |
| **RTX 3090** | ✅ Verified | `bfloat16`, `cuda`, Model Pinning |
| **RTX 4090** | ✅ Verified | Ampere/Ada Tensor Core utilization |
| **CPU** | ⚠️ Fallback | Standard float32 |

### 📈 Impact
Expected **5x to 10x speed increase** over CPU separation on RTX 3090/4090.

## 🧠 Bolt's Philosophy
- **Speed is a feature.**
- **Visual clarity (Emojis) is usability.**
- **Every millisecond and every pixel counts.**

---
*Developed by Bolt ⚡ - Performance-obsessed agent.*
