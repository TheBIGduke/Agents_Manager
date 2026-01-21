# Agents_Manager

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)


Python package for fully offline fuzzy retrieval (fuzzy_search-style) with an optional speech pipeline. Additionally, user can select the "brain" for the agent either local or online by cloning submodules.<br> 
It provides fast, robust approximate matching over a local knowledge base, with improved pattern matching, stronger error handling, and a cleaner, modular codebase. The project is organized by modules (Wake Word, STT, Fuzzy Search, __optionally an agent__, TTS), each living in its own folder with dedicated documentation.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Contributing](#contributing)

---

<h2 id="installation">Installation</h2>

> [!IMPORTANT]
> This implementation was tested on Ubuntu 22.04 with Python 3.10.12

### Prerequisites

- Git, CMake, ffmpeg
- Optional: NVIDIA CUDA for GPU acceleration

### Cloning this Repository

```bash
# Clone the repository
git clone https://springlabsdevs.net/mecatronica/robotica/agents_manager.git
cd agents_manager
```

### Setup

#### For automatic installation and setup, run the installer:

```bash
bash installer.sh
```

#### For manual installation and setup:

```bash
sudo apt update

# General installations
sudo apt install -y python3-dev python3-venv build-essential curl unzip

# STT (Speech-to-Text)
sudo apt install -y portaudio19-dev ffmpeg

# TTS (Text-to-Speech)
# ffmpeg is already installed above
```

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

To verify models were correctly downloaded or to download models:

```bash
.venv/bin/python utils/download.py
```

The script installs everything into your cache directory (`~/.cache/agents_manager`).

---

<h2 id="configuration">Configuration</h2>

> [!WARNING]
> Audio models can be large. Ensure you have enough disk space and RAM/VRAM for your chosen settings.

### General Settings (`config/settings.py`)

All runtime settings are defined in **`config/settings.yml`**. These are plain Python constants—edit the file and restart your scripts to apply changes.

### Model Catalog (`config/models.yml`)

Define which models the system uses (LLM, STT, TTS, wake word) along with their URLs and sample rates.

### Data for Common Questions (`config/data/general_rag.json`)

All general questions and answers are stored in `config/data/general_rag.json`. Define new questions in the `triggers` array and provide the corresponding `answer`.

---

<h2 id="quick-start">Quick Start</h2>

```bash
cd agents_manager
source .venv/bin/activate
```

### Launch the Full Pipeline (Wake Word, STT, Fuzzy Search, TTS)

Start everything with:

```bash
python -m main
```

Now say `ok robot` — the system will start listening and run the complete pipeline.

### Run

**Run the full pipeline (recommended):**

```bash
python -m main
```

**Fuzzy Search Module:**

```bash
python -m fuzzy_search.fuzzy_search
```

**Speech to Text Module:**

```bash
# To test Speech-to-Text 
# Remember to say "ok Robot"
python -m stt.speech_to_text
```

To test wake_word only:
```bash
# Remember to say "ok Robot"
python -m stt.wake_word
```


**Text to Speech Module:**

```bash
python -m tts.text_to_speech
```


> [!TIP]
> If you encounter issues launching modules, try running with the virtual environment explicitly:
> `./.venv/bin/python -m stt.speech_to_text`

---

<h2 id="usage">Usage</h2>

### Fuzzy Search Module

This is a minimal example of what you can do with this package. You will find examples of how to retrieve information from the general knowledge base and respond using the LLM.

> [!TIP]
> When integrating this into your system, consider using the LLM only when truly necessary. In most cases, tasks can be solved with pattern matching or by consuming information from the general knowledge base (fuzzy_search).

### Fuzzy Search Module (fuzzy_search)

This module provides a lightweight, fully offline **fuzzy retrieval layer** over a local knowledge base (`general_rag.json`). It loads trigger/answer pairs at startup, normalizes incoming queries, and then scores each trigger using either **RapidFuzz** (recommended, faster) or Python’s built-in `SequenceMatcher`.

If the best similarity score meets your configured threshold (`fuzzy_logic_accuracy_general`), it returns the corresponding answer; otherwise it returns an empty result.

Configuration is read from `config/settings.yml` under the `fuzzy_search` section (threshold, KB path, and whether to use RapidFuzz). The included CLI example lets you type questions and prints the matched answer when confidence is high enough.


#### Add a New Trigger / Answer

The flow is: **triggers → fuzzy match (fuzzy_search) → answer**.

Your input text is normalized (`norm_text`) before matching (lowercase, accents removed, courtesy words stripped), so keep triggers short and simple.

**Format:** `trigger -> answer`

- `como te llamas -> Mi nombre es Octybot.`
- `quien eres -> Soy un agente virtual con búsqueda difusa en una base de conocimiento local.`
- `que puedes hacer -> Puedo responder preguntas usando una base de conocimiento local (fuzzy_search) sin conexión a internet.`
- `hola -> ¡Hola! ¿En qué puedo ayudarte?`
- `gracias -> ¡De nada!`


##### What you should see

- **fuzzy_search loaded:** Logs like `[Diffuse_Search] Loading GENERAL_QA...` followed by `Loaded 395 fuzzy_search entries`.
- **TTS initialized:** `[TTS] Loading whisper TTS model...` then `Text To Speech initialized.`
- **System ready:** `[System] System Ready & Listening...` plus the banner:
    - `Octybot Virtual Agent`
    - `Say 'Ok Robot' to start...`
    - `Press Ctrl+C to exit`

- **Wake word detected:** `[Wake_Word] Wake word detected: 'okay robot'` and then `Audio sent to STT` (you may also see repeated Partial Match: 'ok robot' lines).
- **STT transcript:** `[STT] STT transcribed = ¿Cómo te llamas?`
- **Fuzzy match (Fuzzy Search):** `[Diffuse_Search] Match: 'como te llamas?' -> 'Mi nombre es Octybot...' (1.00)`

> [!TIP]
> Seeing warnings like **“Performing inference on CPU when CUDA is available”** or the **TBB threading layer** notice is expected in some setups. If you want GPU/STT acceleration, set your STT device to cuda (and make sure your environment supports it).

---

<h2 id="based-on">Based On</h2>

This project is derived from **Agents_Manager-for-Robots** by JossueE. The original repository provides a complete robot voice interaction system including wake word detection, LLM integration, and avatar visualization.

OctyVoice Engine extracts and modernizes the core STT/TTS pipeline with:

- Fully async architecture for better performance
- Streaming TTS for reduced latency
- Enhanced error handling and logging
- Improved device detection
- Simplified API for users who need just voice conversion functionality

For the full system, visit the [original repository](https://github.com/JossueE/Agents_Manager-for-Robots).


---

<h2 id="contributing">Contributing</h2>

Contributions are welcome. Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a Pull Request.

---


