# RAG + Agent Starter

Steps:
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env`
4. `python scripts/ingest.py`
5. `uvicorn app.main:app --reload --port 8000`
