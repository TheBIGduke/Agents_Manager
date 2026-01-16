from typing import Optional
from pathlib import Path

import logging
import numpy as np
import whisper
from difflib import SequenceMatcher

from config.settings  import (
    SAMPLE_RATE_STT, LANGUAGE, SELF_VOCABULARY_STT,DEVICE_SELECTOR_STT,
    NO_SPEECH_THRESHOLD_STT, HALLUCINATION_SILENCE_THRESHOLD_STT) 

class SpeechToText:
    def __init__(self, model_path:str, model_name:str) -> None:
        
        self.log = logging.getLogger("STT")    

        model_path = Path(model_path)

        self.model = whisper.load_model(model_name, download_root = model_path.parent, device=DEVICE_SELECTOR_STT)
        

        # --- This patch is to avoid a bug from Whisper, it helps to catch commonly known hallucination outputs
        # and redirect them to prevent cascading errors and keep the interaction fluid ---
        # Common Whisper hallucinations to filter out
        self.hallucinations = [
            "la universidad",
            "subtítulos realizados por",
            "amara.org",
            "gracias por ver",
            "thanks for watching",
            "suscríbete",
            "dale like",
            "copyright",
            "todos los derechos reservados",
            "hacé clic en el botón",
            "regístrate",
            "la policia"
        ]

    
    def worker_loop(self, audio_bytes: bytes) -> Optional[str | None]:
        """With this we can see if we recieve text or none"""
        if audio_bytes is None:
            return None
        try:
            text = self.stt_from_bytes(audio_bytes)
            if text:
                # Check for Hallucinations
                if self.check_hallucination(text):
                    self.log.warning(f"Hallucination detected: '{text}', Triggering retry")
                    return "**error_audio_retry**" # Magic Key for RAG

                self.log.info(f"STT transcribed = {text}")
                return text
            else:
                # Handle Empty Transcription (Audio detected but no words found)
                self.log.info(f"Empty transcription, Triggering retry")
                return "**error_audio_retry**" 
            
        except Exception as e:
            self.log.error(f"Error in STT module: {e}")
            return None


    def check_hallucination(self, text: str) -> bool:
        """
        Verify if the text is a valid transcription or a hallucination.
        Uses a combination of substring presence and fuzzy matching ratio.
        """
        text_lower = text.lower().strip()
        
        # Check if the hallucination phrase exists in the text
        for h in self.hallucinations:
            if h in text_lower:
                # Calculate similarity ratio
                ratio = SequenceMatcher(None, h, text_lower).ratio()
                
                # If high match (> 0.6) or if it's a repetitive loop (length check)
                if ratio > 0.6 or (len(text_lower) > len(h) * 1.5):
                    return True
                
        # Check for word repetition
        words = text_lower.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    self.log.warning(f"Repetitive loop detected: {words[i]}")
                    return True
                
        return False


    def stt_from_bytes (self, audio_bytes: bytes) -> Optional[str]:
        """
        Convert bytes Int16→tensor float32 normalizado y ejecuta Whisper.
        """
        if not audio_bytes: return None

        # Int16 → float32 [-1, 1]
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return None

        x = pcm.astype(np.float32) / 32768.0

        if SAMPLE_RATE_STT != 16000:
            self.log.warning(f"Whisper only works at 16 Khz, info is being sent at {SAMPLE_RATE_STT}hz")

        result = self.model.transcribe(
            x,
            temperature = (0.0, 0.2, 0.3), # Limit retries to 3 attempts (0.0, 0.2, 0.3), Default (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            fp16=False, 
            language = LANGUAGE, 
            task="transcribe",
            initial_prompt = SELF_VOCABULARY_STT,
            carry_initial_prompt=True,
            condition_on_previous_text = False,
            word_timestamps = True,
            hallucination_silence_threshold = HALLUCINATION_SILENCE_THRESHOLD_STT,
            no_speech_threshold = NO_SPEECH_THRESHOLD_STT,
            compression_ratio_threshold=2.4,
            beam_size=1
            )

        return(result["text"])or None



# ———— Example Usage ————
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] [%(name)s] %(message)s")

    from stt.wake_word import WakeWord
    from stt.audio_listener import AudioListener
    from utils.utils import LoadModel

    model = LoadModel()
    audio_listener = AudioListener()
    ww = WakeWord(str(model.ensure_model("wake_word")[0]))
    stt = SpeechToText(str(model.ensure_model("stt")[0]), model_name="small") #Base = 1 id and "base"
    print(str(model.ensure_model("stt")))
    audio_listener.start_stream()
    
    try:
        print("Este es el nodo de prueba del Speech to Text con Audio Listener y Wake Word \n" \
        "Debes decir La Palabara de activación, es 'ok Robot' - Presione Ctrl+C para salir\n")
        while True:
            result = audio_listener.read_frame(ww.frame_samples)
            n_result = ww.wake_word_detector(result)
            stt.worker_loop(n_result)
    except KeyboardInterrupt:
        audio_listener.terminate()
        print("Saliendo")
        exit(0)