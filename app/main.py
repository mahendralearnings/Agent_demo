from fastapi import FastAPI
from pydantic import BaseModel
from .rag import ingest, rag_answer
from .agent import run_agent

class AskIn(BaseModel):
    question: str

class AgentIn(BaseModel):
    input: str
    session_id: str = "default"

app = FastAPI()

@app.post("/ingest")
def ingest_endpoint():
    return ingest()

@app.post("/ask")
def ask_endpoint(body: AskIn):
    return rag_answer(body.question)

@app.post("/agent")
def agent_endpoint(body: AgentIn):
    return run_agent(body.input, body.session_id)
