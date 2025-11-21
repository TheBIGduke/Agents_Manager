import os
import sys
from contextlib import contextmanager
import pyaudio
from config.settings import AUDIO_LISTENER_DEVICE_ID, AUDIO_LISTENER_SAMPLE_RATE, AUDIO_LISTENER_CHANNELS, AUDIO_LISTENER_FRAMES_PER_BUFFER
import logging

# --- ADD THIS CONTEXT MANAGER ---
@contextmanager
def no_alsa_err():
    """Temporarily suppresses C-level stderr output to silence ALSA warnings."""
    try:
        # Save the original stderr file descriptor
        original_stderr_fd = os.dup(sys.stderr.fileno())
        # Open a null device
        devnull = os.open(os.devnull, os.O_WRONLY)
        # Replace stderr with null
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        # Restore stderr
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        os.close(original_stderr_fd)
        os.close(devnull)
# --------------------------------

def define_device_id(pa:pyaudio.PyAudio = None, preferred:int = AUDIO_LISTENER_DEVICE_ID, log:logging.Logger = None) -> int:
    # ... (Keep the rest of this function exactly as it is) ...
    """ Define the device id to use for audio input."""
    if preferred is not None:
        try:
            return preferred
        except Exception as e:
            log.info(f"Error al usar device_index preferido {preferred}: {e}")
    
    elif pa is None:
        log.warning(f"Pyaudio instance no iniciado, no se puede listar dispositivos.")
        return None

    elif pa is not None:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)   
            if info.get('maxInputChannels', 0) > 0:
                log.info(f"[{i}] {info['name']} (in={info['maxInputChannels']}, rate={int(info.get('defaultSampleRate',0))})")
                if info['name'].lower() == "pulse":
                    log.warning(f"[AudioListener - utils]Usando dispositivo PulseAudio por defecto: {i}")
                    return i

class AudioListener:
    def __init__(self):
        self.log = logging.getLogger("AudioListener")  
        self.sample_rate = AUDIO_LISTENER_SAMPLE_RATE
        
        # --- UPDATE THIS BLOCK ---
        # We wrap the PyAudio initialization with our suppressor
        with no_alsa_err():
            self.audio_interface = pyaudio.PyAudio()
        # -------------------------

        self.device_index = define_device_id(self.audio_interface, AUDIO_LISTENER_DEVICE_ID, self.log)
        self.channels = AUDIO_LISTENER_CHANNELS 
        self.frames_per_buffer = AUDIO_LISTENER_FRAMES_PER_BUFFER
        self.stream = None
        self.log.info(f"AudioListener initialized with device_index={self.device_index}, sample_rate={self.sample_rate}, channels={self.channels}, frames_per_buffer={self.frames_per_buffer} ✅ ")

    # ... (Keep the rest of the class as it is) ...
    def start_stream(self):
        """ Start the audio stream if not already started."""
        if self.stream is None:
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frames_per_buffer,
            )

    def read_frame(self, frame_samples: int) -> bytes:
        """ Read a frame of audio data from the stream."""
        if self.stream is None:
            raise RuntimeError("El Audio stream no se ha comenzado o está fallando la lectura.")
        return self.stream.read(frame_samples, exception_on_overflow=False)

    def stop_stream(self):
        """ Stop the audio stream if it is running."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def terminate(self):   # <-- Make sure you renamed this from 'deleate'
        """ Clean up the audio interface and stream."""
        if self.stream is not None:
            self.stop_stream()
        self.audio_interface.terminate()