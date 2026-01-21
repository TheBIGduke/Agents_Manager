# tts/text_to_speech.py
import torch
import io
import wave
import numpy as np
import pyaudio
import logging
from pathlib import Path
from piper.voice import PiperVoice, SynthesisConfig

# Configuration
from pathlib import Path
import yaml


BASE_DIR = Path(__file__).parent.parent
SETTINGS = BASE_DIR / "config" / "settings.yml"


with SETTINGS.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

device_selector = cfg.get("tts", {}).get("device_id", "cpu")
sample_rate = cfg.get("tts", {}).get("sample_rate", 24000)
volume = cfg.get("tts", {}).get("volume", 2.0)
speed = cfg.get("tts", {}).get("speed", 1.0)
path_to_save = cfg.get("tts", {}).get("path_to_save", "tts/audios")
name_of_outs = cfg.get("tts", {}).get("name_of_outs", "test")
save_wav = cfg.get("tts", {}).get("save_wav", False)


class TTS:
    def __init__(self, model_path:str, model_path_conf:str):
        self.log = logging.getLogger("TTS")
        self.log.info("Loading Whisper TTS model...")
        self.log = logging.getLogger("TTS")
        self.voice = PiperVoice.load(model_path = model_path,config_path = model_path_conf )
        self.sample_rate = sample_rate
        self.count_of_audios = 0
        self.out_path = Path(path_to_save) / Path(name_of_outs) / Path(f"{name_of_outs}_{self.count_of_audios}.wav")
        
        self.syn_config = SynthesisConfig(
            volume = volume,  # half as loud
            length_scale = speed,  # twice as slow
            noise_scale = 1.0,  # more audio variation
            noise_w_scale = 1.0,  # more speaking variation
            normalize_audio=False, # use raw audio from voice
        )

        try:
            self.pa = pyaudio.PyAudio()
        except Exception as e:
            self.log.error(f"Error while trying to start PyAudio: {e}")
            self.pa = None

        self.stream = None
        self.log.info("Text To Speech initialized")

    def synthesize(self, text: str):
        """Convert Text to Speech using Piper, return mono audio float32 [-1,1]"""
        if not text:
            return None
        
        if save_wav:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(self.out_path), "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
                self.count_of_audios += 1
                self.out_path = Path(path_to_save) / Path(name_of_outs) / Path(f"{name_of_outs}_{self.count_of_audios}.wav")

        mem = io.BytesIO()
        with wave.open(mem, "wb") as w:
            # Piper setea params internamente; solo pásale el writer
            self.voice.synthesize_wav(text,w, syn_config=self.syn_config)

        # Lee el WAV del buffer y devuélvelo como float32 normalizado
        mem.seek(0)
        with wave.open(mem, "rb") as r:
            frames = r.readframes(r.getnframes())
            pcm_i16 = np.frombuffer(frames, dtype=np.int16)
        audio_f32 = (pcm_i16.astype(np.float32) / 32768.0)
        return audio_f32

    def play_audio_with_amplitude(self, audio_data, amplitude_callback=None):
        """
        Plays the given float32 numpy array (single-channel).
        If amplitude_callback is provided, pass the amplitude
        of each chunk to it for mouth animation, etc.
        """
        if audio_data is None or len(audio_data) == 0:
            return
        
        # Check if it's a torch Tensor
        if torch.is_tensor(audio_data):
            # Move to CPU if needed, convert to NumPy
            audio_data = audio_data.cpu().numpy()  
            # Now it's a NumPy array, e.g. float32 in [-1..1]

        self.start_stream()

        if self.stream is None:
            self.log.error("Audio streaming service couldn't be started")
            return

        # Convert float32 [-1..1] to int16
        audio_int16 = np.clip(audio_data * 32767.0, -32767.0, 32767.0).astype(np.int16)


        chunk_size = 4096
        idx = 0
        total_frames = len(audio_int16)

        while idx < total_frames:
            chunk_end = min(idx + chunk_size, total_frames)
            chunk = audio_int16[idx:chunk_end]
            try:
                self.stream.write(chunk.tobytes())
            except OSError as e:
                self.log.error(f"Error while writing the audio stream: {e}")
                break

            if amplitude_callback:
                # amplitude = mean absolute value
                amplitude = np.abs(chunk.astype(np.float32)).mean()
                amplitude_callback(amplitude)

            idx += chunk_size

        self.stop_tts()
        return True

    def start_stream(self):
        """ Start the audio stream if not already started."""
        if self.pa is None:
            self.pa = pyaudio.PyAudio()

        if self.stream is None:
            try:
                self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True)
            except Exception as e:
                self.log.error(f"Error while trying to open the output stream: {e}")

    def stop_tts(self):
        """Stop the stream"""

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    def terminate(self):
        """ Call this when shutting down the whole app"""
        self.stop_tts()
        if self.pa is not None:
            self.pa.terminate()
            self.pa = None
            self.log.info("PyAudio ended successfully")



 #———— Example Usage ————
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] [%(name)s] %(message)s")

    from utils.utils import LoadModel

    # Configuration
    from pathlib import Path
    import yaml
    BASE_DIR = Path(__file__).parent.parent
    SETTINGS = BASE_DIR / "config" / "settings.yml"
    with SETTINGS.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    voice = cfg.get("tts", {}).get("voice", 1)

    model = LoadModel()
    voice_id, decoder = model.voice_pair(voice)
    tts = TTS(str(model.ensure_model("tts")[voice_id]), str(model.ensure_model("tts")[decoder]))

    try: 
        print("Este es el script de prueba del Text to Speech - Presione Ctrl+C para salir\n")
        while True:
            text = input("Escribe algo: ")
            get_audio = tts.synthesize(text)
            tts.play_audio_with_amplitude(get_audio)

    except KeyboardInterrupt:
        tts.terminate()
        print(" Saliendo")
        exit(0)