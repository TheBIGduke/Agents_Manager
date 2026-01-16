import logging
from utils.utils import LoadModel, configure_logging
from stt.wake_word import WakeWord
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from llm.llm import LlmAgent
from tts.text_to_speech import TTS
    

class OctybotAgent:
    def __init__(self):
        configure_logging() # <--- Initialize color logging
        self.log = logging.getLogger("System")
        model = LoadModel()
        
        #Speech-to-Text
        self.audio_listener = AudioListener()
        self.wake_word = WakeWord(str(model.ensure_model("wake_word")[0]))
        self.stt = SpeechToText(str(model.ensure_model("stt")[0]), "small") #Other Model "base", id = 1

        #LLM
        self.llm = LlmAgent(model_path = str(model.ensure_model("llm")[0]))

        #Text-to-Speech
        self.tts = TTS(str(model.ensure_model("tts")[0]), str(model.ensure_model("tts")[1]))

        # Start the audio stream
        self.audio_listener.start_stream()
        
        self.log.info("System Ready & Listening...")
    

    def main(self):
        """" This is the state machine logic to work with the system.
            - First you start the Audio Listener Process 
            - Then check if wake_word is detected
            - If is detected you make the stt process
            - Pass this info to the llm
            - The llm split the answers 
            - Publish the answer as tts"""

        text_transcribed = None

        while text_transcribed == None:
            audio_capture = self.audio_listener.read_frame(self.wake_word.frame_samples)
            wake_word_buffer =  self.wake_word.wake_word_detector(audio_capture)
            text_transcribed = self.stt.worker_loop(wake_word_buffer)
            
        for out in self.llm.ask(text_transcribed):
            get_audio = self.tts.synthesize(out)
            self.tts.play_audio_with_amplitude(get_audio)
    
    def stop(self):
        self.audio_listener.terminate()
        self.tts.stop_tts()
        self.log.warning("System Stopped")


 #———— Example Usage ————-
if "__main__" == __name__:
    try:
        llm = OctybotAgent()
        print("\n" + "="*50)
        print(" Octybot Virtual Agent")
        print(" Say 'Ok Robot' to start...")
        print(" Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        while True:
            llm.main() 
    except KeyboardInterrupt:
        llm.stop()
        exit(0)