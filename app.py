# app.py
import os
import traceback
import jwt
import logfire
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI, Request, WebSocket, WebSocketDisconnect, 
    Form, Depends, HTTPException, status, UploadFile, File
)
from fastapi.templating import Jinja2Templates
from openinference.instrumentation.langchain import LangChainInstrumentor

import config
from database import USERS_DB, SECRET_KEY, ALGORITHM, verify_token, get_current_admin
from rate_limiter import WebSocketRateLimiter
from rag_service import (
    run_two_pass_rag, 
    rebuild_vector_store_service, 
    close_vector_client
)

# Initialize Logfire and Instrumentation
logfire.configure()
LangChainInstrumentor().instrument()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_vector_client()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Routes
@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = USERS_DB.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    token = jwt.encode({"sub": username, "role": user["role"]}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "username": username, "role": user["role"]}

@app.get("/")
async def get_homepage(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...), admin: dict = Depends(get_current_admin)):
    docs_dir = getattr(config, "DOCUMENTS_DIR", "./docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    file_path = os.path.join(docs_dir, os.path.basename(file.filename))
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"message": f"Successfully uploaded {file.filename}. Click 'Rebuild' to index it."}

@app.post("/admin/rebuild")
async def rebuild_vector_store(admin: dict = Depends(get_current_admin)):
    rebuild_vector_store_service()
    return {"message": "Qdrant vector store successfully rebuilt and re-indexed."}

async def _safe_send(websocket: WebSocket, text: str):
    try:
        await websocket.send_text(text)
    except Exception:
        pass


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    payload = verify_token(token) if token else None
    if not payload:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user_name = payload.get("sub", "User")
    await websocket.accept()

    limiter = WebSocketRateLimiter(max_requests=3, window_seconds=600)
    history = []  # [(question, answer)] for this connection only

    try:
        await websocket.send_text(
            f"SYSTEM: Welcome {user_name}! Ask me anything (English or Arabic). "
            f"I remember the last {config.HISTORY_TURNS} turns, so follow-up questions work "
            f"(type /reset to clear the memory).\n\n"
        )

        while True:
            user_message = (await websocket.receive_text()).strip()
            if not user_message:
                continue

            if user_message == "/reset":
                history.clear()
                await websocket.send_text("SYSTEM: Conversation memory cleared.\n\n")
                continue

            allowed, retry_after = limiter.is_allowed()
            if not allowed:
                await websocket.send_text(f'RATE_LIMIT:{{"retry_after": {retry_after}}}')
                await websocket.send_text(f"\n[SYSTEM WARNING] Rate limit exceeded! Please wait {retry_after} seconds.\n\n")
                continue

            await websocket.send_text(f"AI ({user_name}): ")
            
            # Delegate heavy RAG operations to the service layer
            try:
                answer = await run_two_pass_rag(user_message, websocket, history)
                if answer:
                    history.append((user_message, answer))
                    del history[:-config.HISTORY_TURNS]
            except WebSocketDisconnect:
                raise
            except Exception as e:
                # keep the connection alive if one question fails
                traceback.print_exc()
                await _safe_send(websocket, f"\n[SERVER ERROR] {type(e).__name__}: {e}\n")

            await _safe_send(websocket, "\n\n")

    except WebSocketDisconnect:
        print("Client disconnected.")