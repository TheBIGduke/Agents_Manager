from pathlib import Path
import os
import yaml
import logging
import sys
import warnings
from typing import Any, Dict, List, TypedDict, Optional
from config.settings import MODELS_PATH

# --- COLOR CODES ---
RESET = "\033[0m"
BOLD = "\033[1m"
GRAY = "\033[90m"

# Level Colors
LEVEL_COLORS = {
    "DEBUG": "\033[37m",     # White
    "INFO": "\033[92m",      # Green
    "WARNING": "\033[93m",   # Yellow
    "ERROR": "\033[91m",     # Red
    "CRITICAL": "\033[41m"   # Red Background
}

# Module Colors (To differentiate sources)
MODULE_COLORS = {
    "Octybot": "\033[38;5;93m",         # Purple
    "Wake_Word": "\033[38;5;213m",      # Pink
    "Speech_To_Text": "\033[38;5;85m",  # Turquoise
    "LLM": "\033[96m",                  # Cyan
    "llm_data": "\033[36m",             # Cyan dim
    "Text-to-Speech": "\033[38;5;178m", # Gold
    "AudioListener": "\033[38;5;208m",  # Orange
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # 1. Timestamp Color
        asctime = self.formatTime(record, self.datefmt)
        
        # 2. Module Name Color
        module_color = MODULE_COLORS.get(record.name, "\033[37m") # Default white
        module_name = f"{module_color}[{record.name}]{RESET}"
        
        # 3. Level Name Color
        level_color = LEVEL_COLORS.get(record.levelname, RESET)
        level_name = f"{level_color}{record.levelname}{RESET}"
        
        # 4. Message Color (Error messages get red text, others standard)
        msg_color = level_color if record.levelno >= logging.WARNING else RESET
        message = f"{msg_color}{record.getMessage()}{RESET}"

        # Format: [TIME] [MODULE] LEVEL: Message
        return f"{GRAY}{asctime}{RESET} {module_name} {level_name}: {message}"

def configure_logging():
    """Sets up the global logging configuration with colors."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(handler)
    
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    # Suppress pkg_resources deprecation warning (often from webrtcvad)
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# --- EXISTING CODE BELOW ---

def load_yaml() -> Dict[str, Any]:
    """Is for load yaml files, but we use it just for models"""
    p = Path(MODELS_PATH)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

class ModelSpec(TypedDict, total=False):
    name: str
    url: str

class LoadModel:
    def __init__(self):
        self.data = load_yaml()

    def extract_section_models(self, section: str) -> List[ModelSpec]:
        "We take the values for the yaml file"
        items = self.data.get(section, [])
        if not isinstance(items, list):
            raise ValueError(f"La sección '{section}' no es una lista")
        
        out: List[ModelSpec] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            out.append({
                "name": item.get("name", ""),
                "url": item.get("url", "")
            })
        return out

    def ensure_model(self, section: str) -> List[Path]:
        """ Ensure the model directory exists, return a List of paths or an error message """
        base_dir = Path.home() / ".cache" / "octy"  
        models = []
        values = self.extract_section_models(section)
        for value in values:     
            model_dir = base_dir/section/ value.get('name')
            if not model_dir.exists():
                raise FileNotFoundError( f"[LLM_LOADER] Ruta directa no existe: {model_dir}\n")
            models.append(Path(model_dir))
        return(models)

if "__main__" == __name__:
    configure_logging()
    ensure_model = LoadModel()
    model = ensure_model.ensure_model("stt")
    print(model[0])