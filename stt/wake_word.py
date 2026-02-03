from __future__ import annotations
import logging, json

import json
import threading
from websocket import create_connection

# --- SILENCE WEBRTCVAD WARNING ---
import warnings
# Suppress the specific pkg_resources warning from webrtcvad
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
import webrtcvad
# ---------------------------------

import vosk
import threading
from collections import deque
from utils.utils import SETTINGS
from utils.utils import send_face_mood

# Configuration
import yaml

with SETTINGS.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

activation_phrase = cfg.get("wake_word", {}).get("activation_phrase", "ok robot")
variants = cfg.get("wake_word", {}).get("variants", ["ok robot", "okay robot", "hey robot"])
vad_aggressiveness = cfg.get("wake_word", {}).get("vad_aggressiveness", 3)
required_hits = cfg.get("wake_word", {}).get("required_hits", 10)
min_silence_ms_to_drain = cfg.get("stt", {}).get("min_silence_ms_to_drain", 100)
listen_seconds = cfg.get("stt", {}).get("listen_seconds", 5)
sample_rate = cfg.get("audio_listener", {}).get("sample_rate", 16000)
channels = cfg.get("audio_listener", {}).get("channels", 1)
debug_mode = cfg.get("debug_mode", False)


class WakeWord:
    def __init__(self, model_path:str, log=None, debug:bool = debug_mode) -> None:

        self.log = log or logging.getLogger("Wake_Word")     
        level = logging.DEBUG if debug else logging.INFO
        self.log.setLevel(level)
        self.wake_word = activation_phrase
        self.listen_seconds = listen_seconds
        self.sample_rate = sample_rate
        self.variants = variants
        grammar = json.dumps(self.variants, ensure_ascii=False)
        
        # --- SILENCE VOSK LOGS ---
        # Sets Vosk C++ library log level to warnings/errors only
        vosk.SetLogLevel(-1) 
        
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate, grammar)

        #Flags
        self.listening_confirm = False
        self.listening = False

        #Debounce parameters 
        self.partial_hits = 0
        self.required_hits = required_hits
        self.silence_frames_to_drain = min_silence_ms_to_drain

        #VAD parameters
        # 10 ms → less latency (160 samples - 16 kHz)
        self.vad = webrtcvad.Vad(vad_aggressiveness)  # Aggressiveness mode
        self.frame_ms = 10
        self.frame_samples = int(self.sample_rate / 1000 * self.frame_ms)  # int16 mono

        #Audio buffer for Output
        self.lock = threading.Lock()
        self.buffer = deque() 
        self.size = 0
        self.max = int(self.listen_seconds * self.sample_rate * channels * 2) #2 bytes per int16 sample
        self.max_2 = int(1 * self.sample_rate * channels * 2) #2 bytes per int16 sample

    def wake_word_detector(self, frame: bytes) -> None | bytes:
        """Process one 10 ms PCM int16 mono frame for wake-word detection."""
        flag = True if self.vad.is_speech(frame, self.sample_rate) else False

        if (self.listening or self.listening_confirm) and flag: #If the system is listening or have a confirmation i save the info
            drained = self.buffer_add(frame)  
            if drained is not None:
                self.log.debug("Max buffer size reached, draining buffer...")
                return drained
        
        if not flag: # If I hear silence
            if self.partial_hits > -self.silence_frames_to_drain:  # Count how much silence is saved
                self.partial_hits -= 1         
            if (self.listening or self.listening_confirm) and self.partial_hits <= -self.silence_frames_to_drain: #If is listening and the voice pass the umbral of silence
                self.partial_hits = 0
                if self.listening_confirm and self.size > 0: # If the wake_word is confirm and something is in the buffer
                    self.log.debug("Wake word and audio are confirm, Sending information...")
                    return self.buffer_drain()
                self.buffer_clear()
                self.log.debug("clearing buffer...")
                return
        
        if self.rec.AcceptWaveform(frame): 
            result = json.loads(self.rec.Result() or "{}")
            text = (result.get("text") or "").lower().strip()
            if text and self.matches_wake(text):
                self.log.info(f"Wake word detected: '{text}'")
                # Listening to the user Mood
                send_face_mood("Escuchando")
                if not self.listening_confirm:           
                    self.listening_confirm = True
                    self.listening = True   
                self.partial_hits = 0
                return
            self.partial_hits = 0 


        else:
            partial = json.loads(self.rec.PartialResult() or "{}").get("partial", "").lower().strip()
            if partial:
                if self.matches_wake(partial): #If something looks like a partial detection     
                    if not self.listening: 
                        self.listening = True
                        # Wake word dectected Mood
                        send_face_mood("Alerta")
                        drained = self.buffer_add(frame) if flag else None
                        if drained is not None:
                            return drained
                    self.partial_hits += 1

                    if self.partial_hits >= self.required_hits:
                        self.log.debug(f"Partial Match: {partial!r}")
                        self.partial_hits = 0
                        return
                else:
                    self.partial_hits = 0
                    send_face_mood("Neutral")

    def buffer_add(self, frame: bytes) -> None | bytes:
        with self.lock:
            self.buffer.append(frame)
            self.size += len(frame)
        if self.size > self.max and self.listening_confirm:
            return self.buffer_drain()
        if self.size > self.max_2 and self.listening and not self.listening_confirm:
            self.log.debug("Detection wasn't confirmed, clearing buffer...")
            self.buffer_clear()
        return None

    def buffer_clear(self) -> None:
        """ Clear the audio buffer and reset flags. """
        self.listening = False
        self.listening_confirm = False
        send_face_mood("Neutral")
        with self.lock:
            self.buffer.clear()
            self.size = 0
    
    def buffer_drain(self) -> bytes:
        """
        Return all buffered audio (as a single bytes object) and clear the buffer.
        Operates atomically under `self.lock`.
        """
        self.log.info("Audio sent to STT")

        with self.lock:
            data = b"".join(self.buffer)
            self.buffer.clear()

        self.size = 0
        self.listening = False
        self.listening_confirm = False
        return data

    def norm(self, s: str) -> str:
        """Normalize string: lowercase, remove accents."""
        s = s.lower()
        return (s.replace("á","a").replace("é","e").replace("í","i")
                .replace("ó","o").replace("ú","u").replace("ü","u"))
    
    def matches_wake(self, text: str) -> bool:
        """ Return True if text matches any variant of the wake word. """
        t = self.norm(text)
        for v in self.variants:
            if self.norm(v) in t:
                return True
        return False



 #———— Example Usage ————
if "__main__" == __name__:
    from utils.utils import configure_logging
    configure_logging()

    from utils.utils import LoadModel
    from stt.audio_listener import AudioListener

    model = LoadModel()
    audio_listener = AudioListener()
    ww = WakeWord(str(model.ensure_model("wake_word")[0]))
    audio_listener.start_stream()

    try: 
        print("Este es el script de prueba del Wake Word con Audio Listener")
        print("La Palabara de activación es 'ok Robot' - Presione Ctrl+C para salir\n")
        while True:
            result = audio_listener.read_frame(ww.frame_samples)
            n_result = ww.wake_word_detector(result)
            if n_result is not None:
                print(f"Wake Word detectada, enviando {len(n_result)} bytes de audio para STT")

    except KeyboardInterrupt:
        audio_listener.terminate()
        print(" Saliendo")
        exit(0)