from typing import Optional
from pathlib import Path

import logging
import numpy as np
import whisper
from difflib import SequenceMatcher

from config.settings  import SAMPLE_RATE_STT, LANGUAGE, SELF_VOCABULARY_STT, DEVICE_SELECTOR_STT

class SpeechToText:
    def __init__(self, model_path:str, model_name:str) -> None:
        
        self.log = logging.getLogger("Speech_To_Text")    

        model_path = Path(model_path)

        self.model = whisper.load_model(model_name, download_root = model_path.parent, device=DEVICE_SELECTOR_STT)

        # List of Whisper common hallucinations
        self.hallucinations = [
            "transcripción vacía"
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
            "No me puedo hablar de lo que es"
        ]

    
    def worker_loop(self, audio_bytes: bytes) -> Optional[str | None]:
        """With this we can see if we recieve text or none"""
        if audio_bytes is None:
            return None
        try:
            text = self.stt_from_bytes(audio_bytes)
            if text:
                if self.check_hallucinations(text):
                    self.log.warning(f"Hallucination detected: '{text}'. Triggering retry")
                    return "**error_audio_retry**" # Key for RAG
                
                self.log.info(f"Se transcribió = {text}")
                return text
            else:
                self.log.info(f"Transcripción vacía")
                return None
            
        except Exception as e:
            self.log.info(f"Error en STT: {e}")


    def check_hallucinations(self, text: str) -> bool:
        """
        Verify if the text is a valid transcription or a hallucination.
        Uses a combination of substring presence and fuzzy matching ratio.
        """
        text_lower = text.lower().strip()

        for h in self.hallucinations:
            if h in text_lower:
                # Calculate similarity ratio
                ratio = SequenceMatcher(None, h, text_lower).ratio()
                if ratio > 0.6 or (len(text_lower) > len(h) * 1.5):
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
            self.log.info(f"Whisper Solo Funciona a 16 Khz, estás enviando información a {SAMPLE_RATE_STT}hz")

        result = self.model.transcribe(
            x,
            temperature = 0.0, 
            fp16=False, 
            language = LANGUAGE, 
            task="transcribe",
            initial_prompt = SELF_VOCABULARY_STT,
            carry_initial_prompt=True,
            condition_on_previous_text = False,
            word_timestamps = True,
            hallucination_silence_threshold = 0.8,
            no_speech_threshold = 0.5,
            compression_ratio_threshold=2.4,
            beam_size=1
            )

        return(result["text"])or None



 #———— Example Usage ————
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

