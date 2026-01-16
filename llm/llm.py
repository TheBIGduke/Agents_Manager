from __future__ import annotations
from typing import List
import logging, json, os

from config.settings import PATH_GENERAL_RAG
from llm.llm_intentions import split_and_prioritize
from llm.llm_data import GENERAL_RAG
from llm.llm_client import LLM
from llm.llm_router import Router


class LlmAgent:
    def __init__(
        self,
        model_path: str,
    ) -> None:
        
        self.log = logging.getLogger("LLM")     
        self.general_rag = GENERAL_RAG(os.path.expanduser(PATH_GENERAL_RAG)) 
        self.llm = LLM(model_path =  model_path)
        self.router = Router(self.llm)
        
        self.log.info("LLM initialized")

    def ask(self, text: str) -> List[str]:
        """ Process a user input:
        - classify into actions (general/rag)
        - execute via router.handle()
        - return list of responses """
        outs: List[str] = []
        if not isinstance(text, str) or not text.strip():
            text = "No tengo mensaje para procesar."
            return [text]
        
        try:
            actions = split_and_prioritize(text, self.general_rag)
            for action in actions:
                data = action.get("params", {}).get("data")
                kind = action.get("kind")
                ans = self.router.handle(data, kind)
                if not isinstance(ans, str):
                    ans = json.dumps(ans, ensure_ascii=False)
                self.log.info(ans)
                outs.append(ans)

        except Exception as e:
            self.log.exception("Error while processing ask()")
            ans = json.dumps({"error": type(e).__name__, "msg": str(e)}, ensure_ascii=False)
        return outs

 #———— Example Usage ————
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] [%(name)s] %(message)s")

    from utils.utils import LoadModel
    model =  LoadModel()
    app = LlmAgent(model_path = str(model.ensure_model("llm")[0]))
    
    print("Prueba de LLM 🤖:")
    print("Escribe una orden - Presiona (Ctrl+C para salir):")
    print("(Ejemplos: '¿Quién eres?', 'Cuéntame un chiste')")
    try:
        while True:
            try:
                text = input("> ").strip()
                if not text:
                    continue
                for out in app.ask(text):
                    print(out)
            except KeyboardInterrupt:
                print("\nPrueba Terminada")
                break
    except Exception:
        logging.exception("Error fatal en el loop principal")