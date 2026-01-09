# Local LLM

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)


Python package that integrates a local Large Language Model with a fully offline speech pipeline, enabling natural language understanding and response without requiring cloud connectivity. This optimized implementation features a streamlined LLM module with improved pattern matching, enhanced error handling, and a cleaner codebase. The system includes everything needed for the complete pipeline: Wake Word Detection, Speech-to-Text (STT), LLM Processing, and Text-to-Speech (TTS). Each module is organized in separate folders with dedicated documentation.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)

---

<h2 id="installation">Installation</h2>

> [!IMPORTANT]
> This implementation was tested on Ubuntu 22.04 with Python 3.10.12

### Prerequisites

- Git, CMake
- Optional: NVIDIA CUDA for GPU acceleration

### Cloning this Repository

```bash
# Clone the repository
git clone https://github.com/TheBIGduke/Local-LLM.git
cd Local-LLM
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

The script installs everything into your cache directory (`~/.cache/octy`).

---

<h2 id="configuration">Configuration</h2>

> [!WARNING]
> LLMs and audio models can be large. Ensure you have enough disk space and RAM/VRAM for your chosen settings.

### General Settings (`config/settings.py`)

All runtime settings are defined in **`config/settings.py`**. These are plain Python constants—edit the file and restart your scripts to apply changes.

### Model Catalog (`config/models.yml`)

Define which models the system uses (LLM, STT, TTS, wake word) along with their URLs and sample rates.

### System Prompt Definition (`config/llm_system_prompt_def.py`)

To rewrite or define a new LLM System Prompt, edit this file.

### Data for Common Questions (`config/data/general_rag.json`)

All general questions and answers are stored in `config/data/general_rag.json`. Define new questions in the `triggers` array and provide the corresponding `answer`.

---

<h2 id="quick-start">Quick Start</h2>

```bash
cd Local-LLM
source .venv/bin/activate
```

### Launch the Full Pipeline (Wake Word, STT, LLM, TTS)

Start everything with:

```bash
python -m main
```

Now say `ok robot` — the system will start listening and run the complete pipeline.

### Run Individual Module Tests

**LLM Module:**

```bash
python -m llm.llm
```

**Speech to Text Module:**

```bash
# To test the Audio Listener 
python -m stt.audio_listener

# To test the Wake Word Detector
python -m stt.wake_word

# To test Speech-to-Text
python -m stt.speech_to_text
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

### LLM Module

This is a minimal example of what you can do with this package. You will find examples of how to retrieve information from the general knowledge base and respond using the LLM.

> [!TIP]
> When integrating this into your system, consider using the LLM only when truly necessary. In most cases, tasks can be solved with pattern matching or by consuming information from the general knowledge base (RAG).

#### Agent Intents (`handle(data, tipo)`)

| `tipo`     | What it does | Input `data` | Output shape | What it represents |
|------------|--------------|--------------|--------------|-------------------|
| `rag`      | Returns `data` as-is (external RAG already resolved). | Pre-composed string from your RAG (e.g., `general_rag.json`) | `str` | Retrieve information from the knowledge base. |
| `general`  | Free-form Q&A via `llm.answer_general`. | Question string | `str` | Minimal example of using the LLM for general queries. |

#### Add a New Intent / Tool

The flow is: **patterns → intent detection → router → tool implementation**.

Your text is normalized (`norm_text`) before matching (lowercase, accents removed, courtesy words stripped), so keep patterns simple.

##### 1) Declare patterns in `llm_patterns.py`

Add a compiled regex that captures the trigger words for your new intent (example: "time" intent).

```python
# llm_patterns.py
import re

TIME_WORDS_RE = re.compile(r"""
(?xi)
\b(
    hora|que\s+hora|time|current\s+time
)\b
""")
```

Add the pattern to the executor:

```python
# Define the functions with corresponding patterns
INTENT_RES = {
    "time": TIME_WORDS_RE,
}

# Define the priority of function execution
INTENT_PRIORITY = ("time",)

# kind_group: "first" (short) or "second" (long) determines execution order
# need_user_input: True if user query is needed, False otherwise
INTENT_ROUTING = {
    "time": {"kind_group": "first", "kind": "time", "need_user_input": False},
}
```

##### 2) Implement the tool in `llm_tools.py` (class GetInfo)

Tools that should be spoken by TTS should return a string.

```python
def tool_get_time(self) -> str:
    from datetime import datetime
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}"
```

##### 3) Wire it into the router

Pass the tool to the `Router` constructor and handle it in `llm_router.py`.

```python
# llm_router.py
class Router:
    def __init__(self, llm):
        self.llm = llm
        self.handlers: Dict[str, Callable[[str], str]] = {
            "rag": self.data_return,
            "general": self.general_response_llm,
            "time": self.publish_time,  # NEW
        }
    
    def publish_time(self, data: str) -> str:  # NEW
        hours, minutes = self.get_info.tool_get_time()
        return f"Son las {hours}:{minutes}"
```

---

#### Avatar

Enable a simple on-screen "avatar" to visualize the pipeline (wake word, STT, LLM, TTS).

##### 1) Turn it on

Open `config/settings.py` and set:

```python
AVATAR = True
```

##### 2) Grant access to your I/O

The avatar uses your microphone and speakers. For setup notes (devices, permissions, troubleshooting), see the [Avatar README](avatar/README.md).

##### 3) Start the avatar

From your project root, either:

**Run the full pipeline (recommended):**

```bash
python -m main
```

Or **run only the Wake Word test (for a quick check):**

```bash
python -m stt.wake_word
```

##### What you should see

- **Idle Mode:** The ball appears blue and remains static with no deformation.
- **Wake word detected:** The ball turns gold, indicating it is actively listening.
- **STT running:** The ball deforms dynamically, following the waveforms of your voice input.
- **TTS speaking:** The ball returns to blue and moves according to the output audio waveform.

> [!TIP]
> If the ball does not deform, you will see a message in the web interface. This usually means there is a configuration issue—please check the [Avatar README](avatar/README.md) for setup details.

This will open a local HTML page (the avatar UI). Try the full flow: say **"ok robot, como te llamas"** and watch the indicators change.

---

<h2 id="based-on">Based On</h2>

This project is derived from **Local-LLM-for-Robots** by JossueE. The original repository provides a complete robot voice interaction system including wake word detection, LLM integration, and avatar visualization.

OctyVoice Engine extracts and modernizes the core STT/TTS pipeline with:

- Fully async architecture for better performance
- Streaming TTS for reduced latency
- Enhanced error handling and logging
- Improved device detection
- Simplified API for users who need just voice conversion functionality

For the full system, visit the [original repository](https://github.com/JossueE/Local-LLM-for-Robots).


<h3 id="short-demo">Short Demo</h3>

Here is a short demo of the Avatar system in action, showing how the visualization reacts when the wake word is detected and triggers the full interaction pipeline.

<p align="center">
  <a href="https://youtu.be/PP4M3LmFDbM" target="_blank">
    <img src="https://img.youtube.com/vi/PP4M3LmFDbM/hqdefault.jpg" width="720" alt="Avatar Demo YouTube">
  </a>
</p>

When the wake word is detected, the avatar changes its color and responds with speech.

> [!NOTE]
> This avatar system was originally developed by [TheBIGduke](https://github.com/TheBIGduke/OctoV)

![Avatar Demo](docs/avatar/avatar.gif)


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