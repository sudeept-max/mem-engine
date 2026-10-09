from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse

from config import Config
from models import MemoryAddRequest, MemoryRecallRequest, RecallResponse
from storage import StorageEngine
from extractor import MemoryExtractor
from contradiction import ContradictionEngine
from retrieval import HybridRetrievalEngine, Embedder

app = FastAPI(title="Local Chatbot Memory Engine")

storage = StorageEngine()
extractor = MemoryExtractor()
embedder = Embedder()
contradiction_engine = ContradictionEngine(storage)
retrieval_engine = HybridRetrievalEngine(storage, embedder)

@app.post("/memory", status_code=201)
def add_memory(req: MemoryAddRequest):
    extracted = extractor.extract(req.message)
    if not extracted:
        return {"status": "ignored", "reason": "Small talk or no storeable facts.", "memories": []}
    saved = contradiction_engine.process_and_store(extracted, embedder)
    return {"status": "success", "memories": saved}

@app.post("/recall", response_model=RecallResponse)
def recall_memories(req: MemoryRecallRequest):
    results = retrieval_engine.recall(req.query, req.top_k)
    return {"memories": results}

@app.get("/memories")
def list_memories(status: str = "ACTIVE"):
    return {"memories": storage.get_all_memories(status)}

@app.delete("/memory/{id}")
def delete_memory(id: int = Path(..., ge=1)):
    success = storage.delete_memory(id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory ID not found.")
    return {"status": "deleted", "id": id}

@app.get("/", response_class=HTMLResponse)
def web_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Local AI Memory Dashboard</title>
        <style>
            body { font-family: -apple-system, sans-serif; margin: 30px; background: #f4f6f8; color: #333; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #1a73e8; }
            input { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            button { background: #1a73e8; color: white; border: none; padding: 10px 18px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #1557b0; }
            .card { background: #f9f9f9; padding: 12px; border-left: 4px solid #1a73e8; margin: 10px 0; }
            .tag { font-size: 0.8em; background: #e0e0e0; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🧠 Chatbot Memory Dashboard</h2>
            <h3>Add Memory Message</h3>
            <input id="msg" placeholder="e.g. My project deadline is October 15.">
            <button onclick="addMem()">Add Memory</button>
            
            <h3>Recall Query</h3>
            <input id="query" placeholder="e.g. When is the project deadline?">
            <button onclick="recallMem()">Query Memory</button>

            <h3>Results / Active Memories</h3>
            <div id="results"></div>
        </div>

        <script>
            async function addMem() {
                let msg = document.getElementById('msg').value;
                let res = await fetch('/memory', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: msg})});
                alert(JSON.stringify(await res.json(), null, 2));
                loadActive();
            }
            async function recallMem() {
                let q = document.getElementById('query').value;
                let res = await fetch('/recall', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: q, top_k: 5})});
                let data = await res.json();
                let html = '<h4>Matches:</h4>';
                data.memories.forEach(m => {
                    html += `<div class="card"><span class="tag">${m.type}</span> <b>${m.text}</b> (Score: ${m.score})</div>`;
                });
                document.getElementById('results').innerHTML = html;
            }
            async function loadActive() {
                let res = await fetch('/memories?status=ACTIVE');
                let data = await res.json();
                let html = '<h4>Active Memories in Store:</h4>';
                data.memories.forEach(m => {
                    html += `<div class="card"><span class="tag">${m.type}</span> [ID: ${m.id}] ${m.text}</div>`;
                });
                document.getElementById('results').innerHTML = html;
            }
            loadActive();
        </script>
    </body>
    </html>
    """