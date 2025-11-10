***

<div align="center">

<a href="https://github.com/Alhibb/omnilingual-asr">
  <img src="./omniASR_header.jpg" alt="Header image with a collage of on-the-ground photos from the transcription gathering efforts in Pakistan and Liberia." width="100%" />
</a>

<br>

<h1 align="center">🌐 Omnilingual ASR</h1>

<p align="center">
  <strong>High-Performance Speech Recognition for Over 1,600 Languages.</strong>
</p>

<p align="center">
    <a href="https://pypi.org/project/omnilingual-asr/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/omnilingual-asr.svg?style=for-the-badge&color=blue"></a>
    <a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge"></a>
    <a href="https://pypi.org/project/omnilingual-asr/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/omnilingual-asr.svg?style=for-the-badge&color=blue"></a>
    <a href="https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/"><img alt="Paper" src="https://img.shields.io/badge/cs.CL-Paper-B31B1B.svg?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/facebook/omniasr-transcriptions"><strong>🤗 Live Demo</strong></a> •
  <a href="http://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition"><strong>Blog Post</strong></a> •
  <a href="https://huggingface.co/datasets/facebook/omnilingual-asr-corpus"><strong>Dataset</strong></a> •
  <a href="#-citation"><strong>Citation</strong></a>
</p>

</div>

---

**Omnilingual ASR** delivers state-of-the-art speech recognition for an unprecedented number of languages. It combines cutting-edge model architectures with a massive, open-source dataset to empower developers and researchers to build more inclusive and powerful voice-enabled applications.

<br>

<table align="center" style="width:100%; border: none;">
  <tr style="background-color: transparent;">
    <td align="center" valign="top" width="33%" style="border: none; padding: 10px;">
      <h3>🌍 Unparalleled Coverage</h3>
      <p>High-quality transcription for 1,600+ languages, with robust zero-shot capabilities to support new and under-resourced languages out of the box.</p>
    </td>
    <td align="center" valign="top" width="33%" style="border: none; padding: 10px;">
      <h3>🚀 Dual Architectures</h3>
      <p>Choose between blazing-fast <strong>CTC models</strong> for real-time applications and high-accuracy <strong>LLM-based models</strong> for ultimate transcription quality.</p>
    </td>
    <td align="center" valign="top" width="33%" style="border: none; padding: 10px;">
      <h3>🌐 Open Ecosystem</h3>
      <p>All code, models, and a massive multilingual speech corpus are released under permissive licenses to accelerate research and innovation.</p>
    </td>
  </tr>
</table>

<div align="center">
  <img src="./result_table.png" alt="Performance results table" width="100%" />
  <p><i>Our 7B LLM-based ASR system achieves SOTA performance, with CER < 10% for 78% of the 1,600+ supported languages.</i></p>
</div>

<details>
<summary><strong>🚀 Navigate the README</strong></summary>

- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Model Zoo](#-model-zoo)
  - [Architectures: CTC vs. LLM](#architectures-ctc-vs-llm)
  - [Model Specifications](#model-specifications)
- [Advanced Usage](#-advanced-usage)
- [Training & Fine-Tuning](#-training--fine-tuning)
- [License & Citation](#-license--citation)

</details>

---

## ⚙️ Installation

First, ensure you have PyTorch and the `libsndfile` library installed.

```bash
# For macOS
brew install libsndfile

# For Debian/Ubuntu
sudo apt-get update && sudo apt-get install libsndfile1
```
*(Windows users may need to follow the [fairseq2 setup guide](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#installing-on-windows).)*

Now, install `omnilingual-asr` from PyPI. We highly recommend using a virtual environment.

```bash
python -m venv venv && source venv/bin/activate
pip install omnilingual-asr
```

## ⚡ Quickstart

Transcribe audio from multiple languages in just a few lines of code. The pipeline automatically handles model downloading, audio processing, and device management (CUDA/MPS/CPU).

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Initialize the pipeline with your chosen model card.
# The 7B LLM model offers the highest quality.
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

# Define your audio files and their corresponding language codes.
audio_files = ["path/to/english_audio.wav", "path/to/german_audio.flac"]
languages = ["eng_Latn", "deu_Latn"]  # {ISO 639-3 code}_{Script}

# Run transcription!
transcriptions = pipeline.transcribe(audio_files, lang=languages, batch_size=2)

# Print the results
for audio_file, text in zip(audio_files, transcriptions):
    print(f"-> {audio_file.split('/')[-1]}: {text}")
```
> **Expected Output:**
> ```
> -> english_audio.wav: The quick brown fox jumps over the lazy dog.
> -> german_audio.flac: Der schnelle braune Fuchs springt über den faulen Hund.
> ```

> **⚠️ Important:** The current inference pipeline is optimized for audio files **under 40 seconds**. Support for long-form audio is planned for a future release.

---

## 🤖 Model Zoo

### Architectures: CTC vs. LLM

Omnilingual ASR offers two model families to fit your specific needs:

| Architecture                  | Best For                                           | Key Characteristics                                     |
| ----------------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| **⚡ CTC** (Connectionist Temporal Classification) | **Speed & Efficiency**<br>_Real-time, large-batch processing_ | Extremely fast inference, low memory/VRAM usage.      |
| **🧠 LLM** (Large Language Model Decoder) | **Accuracy & Quality**<br>_Offline transcription, analysis_ | Highest transcription accuracy, better grammar/punctuation. |


### Model Specifications

Select the model that best matches your hardware and performance requirements. Models are downloaded automatically on first use.

| Family | Model Name & Link                                                               | Parameters | Size   | VRAM¹   | Relative Speed² |
| :----: | :------------------------------------------------------------------------------ | :--------- | :----- | :------ | :-------------- |
| **⚡CTC** | [`omniASR_CTC_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt)      | 325M       | 1.3GB  | ~2GB    | **96x**         |
|        | [`omniASR_CTC_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B.pt)          | 975M       | 3.7GB  | ~3GB    | **48x**         |
|        | [`omniASR_CTC_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B.pt)          | 3.1B       | 12.0GB | ~8GB    | **32x**         |
|        | [`omniASR_CTC_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-7B.pt)          | 6.5B       | 25.0GB | ~15GB   | **16x**         |
| **🧠LLM** | [`omniASR_LLM_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M.pt)      | 1.6B       | 6.1GB  | ~5GB    | ~1x             |
|        | [`omniASR_LLM_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B.pt)          | 2.3B       | 8.5GB  | ~6GB    | ~1x             |
|        | [`omniASR_LLM_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt)          | 4.4B       | 17.0GB | ~10GB   | ~1x             |
|        | [`omniASR_LLM_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B.pt)          | 7.8B       | 30.0GB | ~17GB   | **1x (baseline)** |
|        | [`omniASR_LLM_7B_ZS`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B-ZS.pt)    | 7.8B       | 30.0GB | ~20GB   | ~0.5x           |

*¹ Estimated VRAM for a batch size of 1 with 30s audio in BF16 on an A100 GPU.*
*² Speed relative to the `omniASR_LLM_7B` model.*

<details>
<summary>Click to view SSL foundation models and tokenizers</summary>

- **SSL Models**: `omniASR_W2V_300M`, `omniASR_W2V_1B`, `omniASR_W2V_3B`, `omniASR_W2V_7B`
- **Tokenizers**: `omniASR_tokenizer`, `omniASR_tokenizer_v7` (for LLM_7B models)

</details>

---

## 💡 Advanced Usage

### Evaluate with the Hugging Face Dataset 🤗

Easily test model performance by streaming our [`omnilingual-asr-corpus`](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) dataset directly.

**1. Install data dependencies:**
```bash
pip install "omnilingual-asr[data]"
```

**2. Run the evaluation script:**
```python
from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Load model
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

# Stream a language-specific dataset (e.g., Ligurian)
dataset = load_dataset("facebook/omnilingual-asr-corpus", "lij_Latn", split="train", streaming=True)
batch = next(dataset.iter(batch_size=5))

# Format audio for pipeline
audio_inputs = [{"waveform": item["audio"]["array"], "sample_rate": item["audio"]["sampling_rate"]} for item in batch]

# Transcribe and compare
transcriptions = pipeline.transcribe(audio_inputs, batch_size=5)
for i, (pred, truth) in enumerate(zip(transcriptions, batch["raw_text"])):
    print(f"\nSample {i+1}:")
    print(f"  Ground Truth: {truth}")
    print(f"  Prediction:   {pred}")
```

## 🎓 Training & Fine-Tuning

For users who want to fine-tune a model on their own data, we provide a complete set of tools and recipes.

- **[Data Preparation Guide →](/workflows/dataprep/README.md)**: An end-to-end workflow for preparing multilingual datasets.
- **[Training Recipes →](/workflows/recipes/wav2vec2/asr/README.md)**: Pre-configured recipes to launch training jobs for both CTC and LLM models.

## ⚖️ License & Citation

The code and models are released under the [**Apache 2.0 License**](./LICENSE). The dataset is released under [**CC-BY-4.0**](./LICENSE-CC-BY-4.0.md).

If you use Omnilingual ASR in your work, please cite our paper:
```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} et al.},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```