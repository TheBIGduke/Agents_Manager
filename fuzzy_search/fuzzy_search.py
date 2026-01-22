import json
import logging
from typing import List, Dict, Any
from difflib import SequenceMatcher
from rapidfuzz import fuzz as rf_fuzz
from .normalize_text import norm_text
from utils.utils import SETTINGS

# Configuration
import yaml

with SETTINGS.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

fuzzy_logic_accuracy_general = cfg.get("fuzzy_search", {}).get("fuzzy_logic_accuracy_general", 0.70)
path_general = cfg.get("fuzzy_search", {}).get("path_general", "config/data/general_QA.json")
use_rapidfuzz = cfg.get("fuzzy_search", {}).get("use_rapidfuzz", True)

class GENERAL_QA:
    def __init__(self, path: str):
        self.log = logging.getLogger("Diffuse_Search")
        self.items: List[Dict[str,str]] = []
        self.load(path)
    
    def load(self, path: str) -> None:
        """ Load the GENERAL_QA from a JSON file or line-separated JSON objects """
        self.log.info("Loading GENERAL_QA...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()

            try:
                obj = json.loads(txt)
                items: List[Dict[str,str]] = []
                
                if isinstance(obj, dict):
                    for _, lst in obj.items():
                        if isinstance(lst, list):
                            for it in lst:
                                ans = it.get('answer','')
                                for trig in it.get('triggers',[]):
                                    trig = norm_text(trig, False)
                                    if trig and ans:
                                        items.append({'q': trig, 'a': ans})
                elif isinstance(obj, list):
                    items = obj
                self.items = items
                self.log.info(f"Loaded {len(self.items)} fuzzy_search entries ")
            except json.JSONDecodeError:
                self.items = [json.loads(line) for line in txt.splitlines() if line.strip()]
                self.log.warning("JSON format issue, attempted line-by-line load.")
        except Exception as e:
            self.items = []
            self.log.error(f"Could not open fuzzy_search file: {e}")
    
    def lookup(self, query: str) -> Dict[str, Any]:
        """ Simple exact or fuzzy match in the GENERAL_QA. Returns dict with 'answer' and 'score' (0.0-1.0) """
        if not self.items:
            return {"error":"general_QA_vacia","answer":"","score":fuzzy_logic_accuracy_general}
        query = norm_text(query, False)
        best, best_s = None, 0.0

        for item in self.items:
            q = item.get('q','')
            fuzzy = (rf_fuzz.ratio(query, q)/100.0) if use_rapidfuzz else SequenceMatcher(None, query, q).ratio()
            s = fuzzy
            if s > best_s:
                best, best_s = item, s
        
        if best and best_s >= fuzzy_logic_accuracy_general:
            self.log.info(f"Match: '{query}' -> '{best.get('a','')[:30]}...' ({best_s:.2f})")
            return {"answer": best.get('a',''), "score": round(best_s,3)}
        return {"answer":"","score": round(best_s,3)}
    
    def best_hit(self, res) -> Dict[str, Any]:
        """ From the result of general_rag.lookup (dict or list of dicts), return the best one (highest score)"""
        if isinstance(res, list) and res:
            return max((x for x in res if isinstance(x, dict)), key=lambda x: x.get('score', 0.0), default={})
        return res if isinstance(res, dict) else {}
    

 #———— Example Usage ————
if "__main__" == __name__:
    from utils.utils import SETTINGS
    # Configuration
    import yaml

    with SETTINGS.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    fuzzy_logic_accuracy_general = cfg.get("fuzzy_search", {}).get("fuzzy_logic_accuracy_general", 0.70)
    path_general = cfg.get("fuzzy_search", {}).get("path_general", "config/data/general_QA.json")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] [%(name)s] %(message)s")
    
    print("Prueba de búsqueda difusa:")
    print("Escribe una orden - Presiona (Ctrl+C para salir):")
    print("(Ejemplos: '¿Quién eres?', 'Cuéntame un chiste')")

    app = GENERAL_QA(path_general)

    try:
        while True:
            try:
                text = input("> ").strip()
                if not text:
                    continue
                var = app.best_hit(app.lookup(text))
                if var.get('answer') and var.get('score', 0.0) >= fuzzy_logic_accuracy_general:
                    print(var.get('answer'))
                else:
                    print("No se encontró una respuesta adecuada.")

            except KeyboardInterrupt:
                print("\nPrueba Terminada")
                break
    except Exception:
        logging.exception("Error fatal en el loop principal")
