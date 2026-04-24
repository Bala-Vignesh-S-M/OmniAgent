from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from agent import get_llm, run_react_agent

app = FastAPI(title="Mail Agent Web UI")

# Initialize the LLM once on server startup
print("Initializing Agent LLM...")
llm = get_llm()
print("LLM Ready!")

class ChatRequest(BaseModel):
    query: str

# Serve the static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Takes a user query and returns logs, final answer, and video IDs from the React agent."""
    result = run_react_agent(llm, request.query)
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
