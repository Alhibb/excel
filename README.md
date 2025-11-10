

---

<br>

<div align="center">
  <a href="https://ai.meta.com/">
    <img src="https://img.shields.io/badge/Brought%20to%20you%20by-Meta%20AI-0062E3?style=for-the-badge&logo=meta&logoColor=white" alt="Brought to you by Meta AI">
  </a>
  <br><br>
  <h1>Omnilingual ASR</h1>
  <p><b>Open-Source Multilingual Speech Recognition for 1,600+ Languages</b></p>
</div>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/omnilingual-asr?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/omnilingual-asr/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Read_the-Paper-b31b1b.svg?style=flat-square)](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Demo-yellow.svg?style=flat-square)](https://huggingface.co/spaces/facebook/omnilingual-asr)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Dataset-yellow.svg?style=flat-square)](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)

</div>

<br>

<div align="center">
  <img src="./omniASR_header.jpg" alt="Header image with a collage of on-the-ground photos from the transcription gathering efforts in Pakistan and Liberia." width="100%" />
  <i>Photographs captured during corpus creation efforts in Pakistan and Liberia.</i>
</div>

<br>

**Omnilingual ASR** is a groundbreaking open-source speech recognition system that supports over **1,600 languages**, including hundreds that have never been covered by any ASR technology. By combining scalable zero-shot learning with a flexible model family, Omnilingual ASR makes speech technology radically more inclusive and adaptable for communities, researchers, and developers worldwide.

Our system is designed for broad accessibility, enabling new languages to be added with just a few paired examples—no specialized expertise or massive datasets required.

---

## ✨ Key Features

*   🌍 **Massive Language Coverage:** State-of-the-art transcription for 1,600+ languages and dialects.
*   🚀 **Zero-Shot Capability:** Transcribe languages the model has never been explicitly trained on.
*   ⚙️ **Flexible Model Family:** A suite of models from 300M to 7B parameters (W2V, CTC, and LLM-based) to fit various performance and hardware needs.
*   🤗 **Hugging Face Integration:** Comes with a ready-to-use, large-scale multilingual speech dataset on the Hugging Face Hub.
*   📖 **Open & Accessible:** All models, code, and training recipes are released under the Apache 2.0 license.

<div align="center">
  <img src="./result_table.png" alt="Performance results table" width="100%" />
  <i>Our 7B LLM-ASR system achieves state-of-the-art performance, with a Character Error Rate (CER) below 10% for 78% of the 1,600+ languages evaluated.</i>
</div>

---

## 🚀 Quick Start

Get your first transcription in minutes.

### 1. Prerequisites

*   Python 3.9+
*   `libsndfile` for audio processing.
    *   **macOS:** `brew install libsndfile`
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install libsndfile1`
    *   **Windows:** Requires manual installation; see official documentation.

### 2. Installation

Install the package using your preferred package manager.

<!-- Using tabs for multiple package managers -->
<details>
<summary><b>pip</b></summary>

```bash
pip install omnilingual-asr
```
</details>

<details>
<summary><b>uv</b></summary>

```bash
uv pip install omnilingual-asr
```
</details>

### 3. Transcribe Your First Audio

Create a Python script and use the `ASRInferencePipeline` to transcribe audio files. Models are downloaded automatically on first use.

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Initialize the pipeline with a model card (e.g., the 7B LLM model)
# The model will be downloaded automatically to ~/.cache/fairseq2/
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

# Define your audio files and their corresponding language codes
audio_files = ["/path/to/english_audio.flac", "/path/to/german_audio.wav"]
languages = ["eng_Latn", "deu_Latn"] # ISO 639-3 code + Script

# Run transcription
transcriptions = pipeline.transcribe(audio_files, lang=languages, batch_size=2)

# Print the results
for audio, text in zip(audio_files, transcriptions):
    print(f"Audio: {audio}\nTranscription: {text}\n")

```

> **⚠️ Note on Audio Length:** Currently, the pipeline is optimized for audio files under 40 seconds. Support for chunking and transcribing unlimited-length audio is in development.

---

## 📚 In-Depth Guides

For more advanced use cases, explore our comprehensive documentation.

*   **[Inference Pipeline](./docs/Inference.md):** A deep dive into batch processing, language conditioning, and context examples.
*   **[Supported Languages](./docs/Languages.md):** View the complete list of 1,600+ supported languages and their codes.
*   **[Training & Data Pipeline](./docs/Training.md):** A full guide to preparing datasets and fine-tuning models.
*   **[Model Architecture](./docs/Architecture.md):** Technical details on the W2V, CTC, and LLM model families.

---

## 🤖 Model Zoo

We offer a range of pre-trained models to balance performance and computational cost. The `ASRInferencePipeline` will automatically select the correct tokenizer and configuration based on the chosen model card.

| Model Card | Family | Parameters | Download (FP32) | VRAM¹ | RTF¹ (Speed)² | Key Features |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`omniASR_CTC_300M`** | CTC | 325M | 1.3 GiB | ~2 GiB | **0.001 (96x)** | Lightweight and extremely fast ASR. |
| **`omniASR_CTC_1B`** | CTC | 975M | 3.7 GiB | ~3 GiB | 0.002 (48x) | Balanced speed and accuracy. |
| **`omniASR_CTC_3B`** | CTC | 3.1B | 12.0 GiB | ~8 GiB | 0.003 (32x) | High-accuracy CTC model. |
| **`omniASR_CTC_7B`** | CTC | 6.5B | 25.0 GiB | ~15 GiB | 0.006 (16x) | Maximum-accuracy CTC model. |
| **`omniASR_LLM_300M`** | LLM | 1.6B | 6.1 GiB | ~5 GiB | 0.090 (~1x) | ASR with language conditioning. |
| **`omniASR_LLM_1B`** | LLM | 2.3B | 8.5 GiB | ~6 GiB | 0.091 (~1x) | - |
| **`omniASR_LLM_3B`** | LLM | 4.4B | 17.0 GiB | ~10 GiB | 0.093 (~1x) | - |
| **`omniASR_LLM_7B`** | LLM | 7.8B | 30.0 GiB | ~17 GiB | **0.092 (1x)** | **Recommended model for best performance.** |
| **`omniASR_LLM_7B_ZS`** | LLM | 7.8B | 30.0 GiB | ~20 GiB | 0.194 (~0.5x) | Specialized for Zero-Shot tasks. |

<p align="right"><i>¹ VRAM and Real-Time Factor (RTF) measured on an A100 GPU with BF16, batch size 1, and 30s audio.</i></p>
<p align="right"><i>² Speed is relative to the `omniASR_LLM_7B` model.</i></p>

---

## 📈 Using the Hugging Face Dataset

We provide a large-scale multilingual speech corpus, `facebook/omnilingual-asr-corpus`, under a CC-BY-4.0 license. You can use it directly for evaluation.

First, install the necessary data-handling libraries:
```bash
pip install "omnilingual-asr[data]"
```

Then, load the dataset and run inference:
```python
from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# 1. Load the dataset for a specific language (e.g., Ligurian - "lij_Latn")
dataset = load_dataset("facebook/omnilingual-asr-corpus", "lij_Latn", split="train", streaming=True)
batch = next(dataset.iter(batch_size=5))

# 2. Format the audio data for the pipeline
audio_inputs = [
    {"waveform": sample["array"], "sample_rate": sample["sampling_rate"]}
    for sample in batch["audio"]
]

# 3. Initialize the pipeline and transcribe
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
transcriptions = pipeline.transcribe(audio_inputs, batch_size=5)

# 4. Compare predicted text with ground truth
for i, (predicted, truth) in enumerate(zip(transcriptions, batch["raw_text"])):
    print(f"--- Sample {i+1} ---")
    print(f"  Ground Truth: {truth}")
    print(f"  Predicted:    {predicted}\n")

```

---

##  License

The Omnilingual ASR codebase and its pre-trained models are licensed under the [Apache 2.0 License](./LICENSE).

## Citing Omnilingual ASR

If you use our models or code in your research, please cite our work. An arXiv version of the paper will be available shortly.

```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} and Keren, Gil and Kozhevnikov, Artyom and Meng, Yen and Ropers, Christophe and Setzler, Matthew and Wang, Skyler and Adebara, Ife and Auli, Michael and Chan, Kevin and Cheng, Chierh and Chuang, Joe and Droof, Caley and Duppenthaler, Mark and Duquenne, Paul-Ambroise and Erben, Alexander and Gao, Cynthia and Mejia Gonzalez, Gabriel and Lyu, Kehan and Miglani, Sagar and Pratap, Vineel and Sadagopan, Kaushik Ram and Saleem, Safiyyah and Turkatenko, Arina and Ventayol-Boada, Albert and Yong, Zheng-Xin and Chung, Yu-An and Maillard, Jean and Moritz, Rashel and Mourachko, Alexandre and Williamson, Mary and Yates, Shireen},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```

<br>
<div align="center">
  <a href="https://ai.meta.com">
    <img src="https://github.com/facebookresearch/fairseq/blob/main/docs/assets/logo-dark.png?raw=true" width="150">
  </a>
</div>