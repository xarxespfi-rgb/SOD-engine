from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="SOD Pedagògic Engine")

# Models de dades
class EvaluationRequest(BaseModel):
    context: str
    observations: List[str]
    evidences: List[str]
    hypothesis: str

class MemoryRecord(BaseModel):
    level: str  # MACRO, MESO, MICRO
    context_tags: List[str]
    learning_point: str

# Memòria temporal en RAM (es pot connectar a Supabase més endavant)
memory_db = []

@app.get("/")
def home():
    return {"status": "SOD Engine actiu i operatiu"}

@app.post("/evaluate-hypothesis")
def evaluate_hypothesis(data: EvaluationRequest):
    # Lògica de Reasoning Policy: Ponderació d'evidències vs observacions
    num_obs = len(data.observations)
    num_evid = len(data.evidences)
    
    if num_evid == 0:
        confidence = "BAIXA"
        recommendation = "Manca d'evidències empíriques. Cal formular preguntes discriminants."
    elif num_evid >= num_obs:
        confidence = "ALTA"
        recommendation = "Hipòtesi ben fonamentada. Es pot procedir a la presa de decisió."
    else:
        confidence = "MITJANA"
        recommendation = "Evidència parcial. Es recomana validar les observacions restants."

    return {
        "hypothesis": data.hypothesis,
        "confidence_level": confidence,
        "recommendation": recommendation,
        "metrics": {
            "observations_count": num_obs,
            "evidences_count": num_evid
        }
    }

@app.post("/store-memory")
def store_memory(record: MemoryRecord):
    memory_db.append(record.dict())
    return {"status": "success", "total_records": len(memory_db)}

@app.get("/query-memory")
def query_memory(level: Optional[str] = None):
    if level:
        filtered = [r for r in memory_db if r["level"].upper() == level.upper()]
        return {"records": filtered}
    return {"records": memory_db}
