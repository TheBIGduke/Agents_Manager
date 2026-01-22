import logging
from utils.utils import LoadModel, configure_logging
from stt.wake_word import WakeWord
from stt.audio_listener import AudioListener
from stt.speech_to_text import SpeechToText
from fuzzy_search.fuzzy_search import GENERAL_QA
from tts.text_to_speech import TTS
from utils.utils import SETTINGS

# Configuration
import yaml
import sys
from pathlib import Path

with SETTINGS.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

fuzzy_logic_accuracy_general = cfg.get("fuzzy_search", {}).get("fuzzy_logic_accuracy_general", 0.70)
path_general = cfg.get("fuzzy_search", {}).get("path_general", "config/data/general_rag.json")
voice = cfg.get("tts", {}).get("voice", 1)
local_llm_path = Path(cfg.get("llm", {}).get("repo_path_local_llm", "~/Local-LLM")).expanduser().resolve()
sys.path.insert(0, str(local_llm_path))

agent_selector = cfg.get("llm", {}).get("agent_selector", "only_fuzzy")  # "local" or "internet" or "only_fuzzy"


class OctybotAgent:
    def __init__(self):
        configure_logging() # <--- Initialize color logging
        self.log = logging.getLogger("System")
        model = LoadModel()
        
        #Speech-to-Text
        self.audio_listener = AudioListener()
        self.wake_word = WakeWord(str(model.ensure_model("wake_word")[0]))
        self.stt = SpeechToText(str(model.ensure_model("stt")[1]), "small") #Other Model "base", id = 1
        print(f"STT Model Loaded: {model.ensure_model('stt')[1]}")
        #Fuzzy Search for fuzzy_search
        self.diff = GENERAL_QA(path_general)

        #Text-to-Speech
        voice_id, decoder = model.voice_pair(voice)
        self.tts = TTS(str(model.ensure_model("tts")[voice_id]), str(model.ensure_model("tts")[decoder]))

        # Start the audio stream
        self.audio_listener.start_stream()   

        self.agent_selector = model.select_agent(agent_selector, cfg, self.log)   

        if self.agent_selector == "local":
            from llm_main import LlmAgent
            self.llm_agent = LlmAgent(model_path = str(model.ensure_model("llm")[0]))
            self.log.info("Local LLM Agent loaded successfully.") 
                               
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

        out = self.diff.best_hit(self.diff.lookup(text_transcribed))
        
        if out.get('answer') and out.get('score', 0.0) >= fuzzy_logic_accuracy_general:
            out = out.get('answer')
            get_audio = self.tts.synthesize(out)
            self.tts.play_audio_with_amplitude(get_audio)

        # IMPORTANT:  In this case the exception "else" is added in the main, so it  gives flexibility to add custom next steps to the system.
        # Considering that this let you work as a state machine, so for example, if you want to the LLM that works with internet,
        # you can add the next steps without modifying the core system.

        elif self.agent_selector == "local":

            for out in self.llm_agent.ask(text_transcribed):
                get_audio = self.tts.synthesize(out)
                self.tts.play_audio_with_amplitude(get_audio)
                                               
        else:
            get_audio = self.tts.synthesize("No se encontró una respuesta adecuada")
            self.tts.play_audio_with_amplitude(get_audio)
            self.log.info("No se encontró una respuesta adecuada.")

    
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
