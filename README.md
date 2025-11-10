***

<div align="center">
  <img src="./omniASR_header.jpg" alt="Header image with a collage of on-the-ground photos from the transcription gathering efforts in Pakistan and Liberia." width="100%" />
  <p><i>Photographs captured during corpus creation efforts in Pakistan and Liberia.</i></p>
</div>

<div align="center">

# Omnilingual ASR

### State-of-the-Art Speech Recognition for 1,600+ Languages

[![PyPI Version](https://img.shields.io/pypi/v/omnilingual-asr.svg?style=flat-square)](https://pypi.org/project/omnilingual-asr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/omnilingual-asr.svg?style=flat-square)](https://pypi.org/project/omnilingual-asr/)
[![Paper](https://img.shields.io/badge/cs.CL-arXiv:2501.XXXXX-B31B1B.svg?style=flat-square)](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/)

</div>

**Omnilingual ASR** unlocks high-quality speech recognition for over 1,600 languages, including hundreds never before supported by any ASR system. It's built to be accessible, scalable, and adaptable, empowering researchers and developers to bring speech technology to communities worldwide.

<div align="center">

[**🤗 HuggingFace Demo**](https://huggingface.co/spaces/facebook/omniasr-transcriptions) &nbsp;&nbsp;•&nbsp;&nbsp;
[**📖 Paper**](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/) &nbsp;&nbsp;•&nbsp;&nbsp;
[**📝 Blog Post**](http://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition) &nbsp;&nbsp;•&nbsp;&nbsp;
[**💾 Dataset**](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)

</div>

---

<div align="center">
  <img src="./result_table.png" alt="Performance results table" width="100%" />
  <p><i>Our 7B LLM-based ASR system achieves state-of-the-art performance, with character error rates (CER) below 10% for 78% of the 1,600+ supported languages.</i></p>
</div>

## ✨ Key Features

*   ✅ **Unprecedented Language Coverage:** State-of-the-art performance on **1,600+ languages**, with robust zero-shot capabilities for adding new ones.
*   🚀 **Flexible Model Families:** Choose between lightning-fast **CTC** models for real-time applications and high-accuracy **LLM-based** models for ultimate quality.
*   📦 **Simple & Powerful API:** A clean, easy-to-use inference pipeline that handles batching, device placement, and language conditioning automatically.
*   🔧 **Built on a Research-Grade Toolkit:** Powered by [fairseq2](https://github.com/facebookresearch/fairseq2), providing a robust foundation for training and extension.
*   🌐 **Fully Open Source:** All code, models, and a massive multilingual dataset are released under permissive licenses ([Apache 2.0](./LICENSE) and [CC-BY-4.0](./LICENSE-CC-BY-4.0.md)).

## 📜 Table of Contents

*   [Getting Started](#-getting-started)
    *   [Installation](#installation)
    *   [Quickstart: Your First Transcription](#quickstart-your-first-transcription)
*   [Advanced Usage](#-advanced-usage)
    *   [Transcribing from the Hugging Face Dataset](#transcribing-from-the-hugging-face-dataset-)
    *   [Listing Supported Languages](#listing-supported-languages)
*   [Model Zoo](#-model-zoo)
    *   [Understanding the Architectures](#understanding-the-architectures)
    *   [Model Specifications](#model-specifications)
    *   [Model Management](#model-management)
*   [Training & Fine-Tuning](#-training--fine-tuning)
*   [Community & Contributing](#-community--contributing)
*   [License & Citation](#-license--citation)

## 🚀 Getting Started

### Installation

Omnilingual ASR is built on PyTorch and [fairseq2](https://github.com/facebookresearch/fairseq2). For full audio support, you'll need `libsndfile`.

```bash
# macOS
brew install libsndfile

# Ubuntu/Debian
sudo apt-get install libsndfile1
```
*(Windows users may require additional [setup steps](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#installing-on-windows) for audio libraries.)*

Now, install the library from PyPI:
```bash
# Recommended: Use a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install using pip
pip install omnilingual-asr

# Or with the lightning-fast `uv`
uv pip install omnilingual-asr
```

### Quickstart: Your First Transcription

Transcribing audio is designed to be simple. The `ASRInferencePipeline` handles model loading, audio processing, and batching.

```python
import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# The pipeline automatically selects the best available device (CUDA, MPS, or CPU)
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

# Prepare your inputs
audio_paths = ["/path/to/english_audio.wav", "/path/to/german_audio.flac"]
languages = ["eng_Latn", "deu_Latn"] # ISO 639-3 code + script

# Transcribe!
transcriptions = pipeline.transcribe(audio_paths, lang=languages, batch_size=2)

# See the results
for path, text in zip(audio_paths, transcriptions):
    print(f"{path}: {text}")
```
> **Expected Output:**
> ```
> /path/to/english_audio.wav: The quick brown fox jumps over the lazy dog.
> /path/to/german_audio.flac: Der schnelle braune Fuchs springt über den faulen Hund.
> ```

> **⚠️ Note on Audio Length:** The current inference pipeline supports audio files up to **40 seconds**. Support for long-form transcription via chunking is coming soon!

---

## 💡 Advanced Usage

### Transcribing from the Hugging Face Dataset 🤗

You can easily evaluate our models on the `omnilingual-asr-corpus` dataset available on Hugging Face. First, install the required data dependencies:

```bash
pip install "omnilingual-asr[data]"
```

Then, stream the dataset and run inference directly on the audio data:

```python
from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# 1. Load the model
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

# 2. Stream a dataset split (e.g., Ligurian)
dataset = load_dataset("facebook/omnilingual-asr-corpus", "lij_Latn", split="train", streaming=True)
batch = next(dataset.iter(batch_size=5))

# 3. Format the audio for the pipeline
audio_inputs = [{"waveform": item["audio"]["array"], "sample_rate": item["audio"]["sampling_rate"]} for item in batch]

# 4. Transcribe
transcriptions = pipeline.transcribe(audio_inputs, batch_size=5)

# 5. Compare results
for i, (pred, truth) in enumerate(zip(transcriptions, batch["raw_text"])):
    print(f"Sample {i+1}:")
    print(f"  GT:  {truth}")
    print(f"  Pred: {pred}\n")
```

### Listing Supported Languages

Languages are identified by their `{ISO 639-3 code}_{Script}`. You can programmatically access the full list:
```python
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

print(f"Total supported languages: {len(supported_langs)}")
# > Total supported languages: 1640

# Check for a specific language
if "spa_Latn" in supported_langs:
    print("Spanish (Latin script) is supported!")
```

## 🤖 Model Zoo

### Understanding the Architectures

We provide two main families of ASR models, each with different trade-offs:

*   **CTC Models (`omniASR_CTC_*`)**: These models use a Connectionist Temporal Classification decoder. They are **extremely fast** and memory-efficient, making them ideal for real-time or large-batch processing where latency is critical.
*   **LLM-based Models (`omniASR_LLM_*`)**: These models couple the audio encoder with a powerful Large Language Model decoder. They offer the **highest accuracy** and better handle complex grammar and punctuation, at the cost of higher computational requirements.

### Model Specifications

The table below details the available pretrained models. Select the one that best fits your accuracy, speed, and hardware constraints.

| Model Family | Model Name & Link                                                               | Parameters | Size   | VRAM¹   | Relative Speed² | Key Features                               |
| :----------- | :------------------------------------------------------------------------------ | :--------- | :----- | :------ | :-------------- | :----------------------------------------- |
| **CTC**      | [`omniASR_CTC_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt)      | 325M       | 1.3GB  | ~2GB    | **96x**         | Blazing fast, low resource ASR             |
|              | [`omniASR_CTC_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B.pt)          | 975M       | 3.7GB  | ~3GB    | **48x**         |                                            |
|              | [`omniASR_CTC_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B.pt)          | 3.1B       | 12.0GB | ~8GB    | **32x**         |                                            |
|              | [`omniASR_CTC_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-7B.pt)          | 6.5B       | 25.0GB | ~15GB   | **16x**         |                                            |
| **LLM**      | [`omniASR_LLM_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M.pt)      | 1.6B       | 6.1GB  | ~5GB    | ~1x             | High-quality ASR with language conditioning |
|              | [`omniASR_LLM_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B.pt)          | 2.3B       | 8.5GB  | ~6GB    | ~1x             |                                            |
|              | [`omniASR_LLM_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt)          | 4.4B       | 17.0GB | ~10GB   | ~1x             |                                            |
|              | [`omniASR_LLM_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B.pt)          | 7.8B       | 30.0GB | ~17GB   | **1x (baseline)** | SOTA quality with language conditioning     |
|              | [`omniASR_LLM_7B_ZS`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B-ZS.pt)    | 7.8B       | 30.0GB | ~20GB   | ~0.5x           | Optimized for Zero-Shot ASR                |

<details>
<summary><b>Self-Supervised & Tokenizer Models</b></summary>

| Model Family | Model Name & Link                                                                 | Parameters | Size   | Features                                           |
| :----------- | :-------------------------------------------------------------------------------- | :--------- | :----- | :------------------------------------------------- |
| **SSL**      | [`omniASR_W2V_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt)        | 317M       | 1.2GB  | Self-Supervised Learning encoder backbone          |
|              | [`omniASR_W2V_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-1B.pt)            | 965M       | 3.6GB  |                                                    |
|              | [`omniASR_W2V_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-3B.pt)            | 3.1B       | 12.0GB |                                                    |
|              | [`omniASR_W2V_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-7B.pt)            | 6.5B       | 25.0GB |                                                    |
| **Tokenizer**| [`omniASR_tokenizer`](https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model)   | -          | 100KB  | Tokenizer for all models except LLM_7B             |
|              | [`omniASR_tokenizer_v7`](https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model) | -          | 100KB  | Tokenizer for `omniASR_LLM_7B` models              |

</details>

*¹ Inference VRAM is an estimate for a batch size of 1 with 30s of audio in BF16 on an A100 GPU.*
*² Relative speed is measured against the `omniASR_LLM_7B` model as a baseline.*

### Model Management

-   **Automatic Caching:** Models and assets are automatically downloaded and cached on their first use.
-   **Storage Location:** By default, assets are stored in `~/.cache/fairseq2/assets/`. You can learn more about fairseq2's [Asset Store System](https://facebookresearch.github.io/fairseq2/stable/basics/assets.html#the-asset-store-system).

## 🎓 Training & Fine-Tuning

For researchers and developers looking to fine-tune our models on custom datasets, we provide comprehensive guides and recipes:

1.  **Data Preparation:** Follow our [end-to-end guide](/workflows/dataprep/README.md) for preparing multilingual datasets, including integration with Hugging Face and processing Parquet files.
2.  **Training Recipes:** Use our pre-configured workflows in the [recipes directory](/workflows/recipes/wav2vec2/asr/README.md) to launch training jobs for both CTC and LLM model families.

## 🤝 Community & Contributing

We welcome contributions from the community! Whether it's reporting a bug, improving documentation, or adding a new feature, your help is valued.

*   **GitHub Issues:** The best place for bug reports, feature requests, and discussions is our [Issues page](https://github.com/Alhibb/omnilingual-asr/issues).
*   **Contributing:** Please see our (upcoming) `CONTRIBUTING.md` for guidelines on how to contribute to the project.

## ⚖️ License & Citation

The code and models in this repository are released under the [Apache 2.0 License](./LICENSE). The dataset is released under [CC-BY-4.0](./LICENSE-CC-BY-4.0.md).

If you use Omnilingual ASR in your research, please cite our work:

```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} and Keren, Gil and Kozhevnikov, Artyom and Meng, Yen and Ropers, Christophe and Setzler, Matthew and Wang, Skyler and Adebara, Ife and Auli, Michael and Chan, Kevin and Cheng, Chierh and Chuang, Joe and Droof, Caley and Duppenthaler, Mark and Duquenne, Paul-Ambroise and Erben, Alexander and Gao, Cynthia and Mejia Gonzalez, Gabriel and Lyu, Kehan and Miglani, Sagar and Pratap, Vineel and Sadagopan, Kaushik Ram and Saleem, Safiyyah and Turkatenko, Arina and Ventayol-Boada, Albert and Yong, Zheng-Xin and Chung, Yu-An and Maillard, Jean and Moritz, Rashel and Mourachko, Alexandre and Williamson, Mary and Yates, Shireen},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```