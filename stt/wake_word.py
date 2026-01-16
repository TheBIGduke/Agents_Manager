from __future__ import annotations
import logging, json

# --- SILENCE WEBRTCVAD WARNING ---
import warnings
# Suppress the specific pkg_resources warning from webrtcvad
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
import webrtcvad
# ---------------------------------

import vosk
import threading
from collections import deque

from config.settings import (
    MIN_SILENCE_MS_TO_DRAIN_STT, ACTIVATION_PHRASE_WAKE_WORD,
    LISTEN_SECONDS_STT, AUDIO_LISTENER_SAMPLE_RATE,
    VARIANTS_WAKE_WORD, AUDIO_LISTENER_CHANNELS,
    AVATAR, VAD_AGGRESSIVENESS, WAKE_WORD_REQUIRED_HITS
)

if AVATAR:
    import webbrowser, subprocess, sys
    from pathlib import Path
    from avatar.avatar_server import send_mode_sync


class WakeWord:
    def __init__(self, model_path:str) -> None:

        self.log = logging.getLogger("Wake_Word")     
        self.wake_word = ACTIVATION_PHRASE_WAKE_WORD
        self.listen_seconds = LISTEN_SECONDS_STT
        self.sample_rate = AUDIO_LISTENER_SAMPLE_RATE
        self.variants = VARIANTS_WAKE_WORD
        
        #State Machine 
        # --- CHANGED FROM DEBUG TO INFO ---
        self.on_say = (lambda s: self.log.info(f"{s}"))
        # ----------------------------------

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
        self.required_hits = WAKE_WORD_REQUIRED_HITS
        self.silence_frames_to_drain = MIN_SILENCE_MS_TO_DRAIN_STT

        #VAD parameters
        # 10 ms → less latency (160 samples - 16 kHz)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)  # Aggressiveness mode
        self.frame_ms = 10
        self.frame_samples = int(self.sample_rate / 1000 * self.frame_ms)  # int16 mono

        #Audio buffer for Output
        self.lock = threading.Lock()
        self.buffer = deque() 
        self.size = 0
        self.max = int(self.listen_seconds * self.sample_rate * AUDIO_LISTENER_CHANNELS * 2) #2 bytes per int16 sample
        self.max_2 = int(1 * self.sample_rate * AUDIO_LISTENER_CHANNELS * 2) #2 bytes per int16 sample

        #Initialize Avatar Server if needed
        if AVATAR:
            subprocess.Popen([sys.executable, "-m", "avatar.avatar_server"], stdin=subprocess.DEVNULL, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
            webbrowser.open(Path("avatar/OctoV.html").resolve().as_uri(), new=0, autoraise=True)

    def wake_word_detector(self, frame: bytes) -> None | bytes:
        """Process one 10 ms PCM int16 mono frame for wake-word detection."""
        flag = True if self.vad.is_speech(frame, self.sample_rate) else False

        if (self.listening or self.listening_confirm) and flag: #If I'm listening or If I got a confirmation i save the info
            drained = self.buffer_add(frame)  
            if drained is not None:
                send_mode_sync(mode = "TTS", as_json=False) if AVATAR else None
                return drained
        
        if not flag: # If I hear silence
            if self.partial_hits > -self.silence_frames_to_drain:  # I count how much silence I have
                self.partial_hits -= 1         
            if (self.listening or self.listening_confirm) and self.partial_hits <= -self.silence_frames_to_drain: #If I'm listening and I pass my umbral of silence
                self.partial_hits = 0
                send_mode_sync(mode = "TTS", as_json=False) if AVATAR else None
                if self.listening_confirm and self.size > 0: # If I have the wake_word comfirm and I have something
                    return self.buffer_drain()
                self.on_say("Detection wasn't confirmed, clearing buffer...")
                self.buffer_clear()
                return
        
        if self.rec.AcceptWaveform(frame): 
            result = json.loads(self.rec.Result() or "{}")
            text = (result.get("text") or "").lower().strip()
            if text and self.matches_wake(text):
                self.log.info(f"Wake word detected: '{text}'")
                if not self.listening_confirm:           
                    self.listening_confirm = True
                    self.listening = True   
                self.partial_hits = 0
                return
            self.partial_hits = 0

        else:
            partial = json.loads(self.rec.PartialResult() or "{}").get("partial", "").lower().strip()
            if partial:
                if self.matches_wake(partial): #If I got something that looks like partial     
                    if not self.listening: 
                        self.listening = True
                        send_mode_sync(mode = "USER", as_json=False) if AVATAR else None
                        drained = self.buffer_add(frame) if flag else None
                        if drained is not None:
                            return drained
                    self.partial_hits += 1

                    if self.partial_hits >= self.required_hits:
                        self.log.info(f"Partial Match: {partial!r}")
                        self.partial_hits = 0
                        return
                else:
                    self.partial_hits = 0

    
    def buffer_add(self, frame: bytes) -> None | bytes:
        with self.lock:
            self.buffer.append(frame)
            self.size += len(frame)
        if self.size > self.max and self.listening_confirm:
            return self.buffer_drain()
        if self.size > self.max_2 and self.listening and not self.listening_confirm:
            self.on_say("Límite de tiempo alcanzado sin confirmación, limpiando buffer")
            self.buffer_clear()
            send_mode_sync(mode = "TTS", as_json=False) if AVATAR else None
        return None

    def buffer_clear(self) -> None:
        """ Clear the audio buffer and reset flags. """
        self.listening = False
        self.listening_confirm = False
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
        print("Este es el nodo de prueba del Wake Word con Audio Listener")
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