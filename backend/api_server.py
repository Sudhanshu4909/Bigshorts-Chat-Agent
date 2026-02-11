# FastAPI Wrapper for BigShorts LangChain Agent
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from advanced_bigshorts_agent import AdvancedBigShortsAgent

# Initialize FastAPI
app = FastAPI(
    title="BigShorts AI Assistant API",
    description="LangChain-powered autonomous agent for BigShorts platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (singleton)
agent = None

# Request/Response models
class QueryRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    response: Dict[Any, Any]
    session_id: str

class SessionHistory(BaseModel):
    session_id: str
    history: list

class AnalyticsResponse(BaseModel):
    analytics: Dict[Any, Any]

# Startup event
@app.on_event("startup")
async def startup_event():
    global agent
    model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    agent = AdvancedBigShortsAgent(model_path, enable_rag=True)
    print("BigShorts Agent initialized and ready!")

# Health check
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "BigShorts AI Assistant",
        "version": "2.0.0",
        "features": ["RAG", "Multi-Session", "Analytics"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_ready": agent is not None}

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        response = agent.process_query(request.message, request.session_id)
        return QueryResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get session history
@app.get("/session/{session_id}/history", response_model=SessionHistory)
async def get_history(session_id: str):
    """Get conversation history for a session"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        history = agent.get_session_history(session_id)
        return SessionHistory(session_id=session_id, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clear session
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get analytics
@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get usage analytics"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        analytics = agent.get_analytics()
        return AnalyticsResponse(analytics=analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Content guide endpoint
@app.get("/content-types")
async def get_content_types():
    """Get available content types"""
    from advanced_bigshorts_agent import ALLOWED_CONTENT_TYPES
    return {
        "content_types": ALLOWED_CONTENT_TYPES,
        "total": len(ALLOWED_CONTENT_TYPES)
    }

# Issue types endpoint
@app.get("/issue-types")
async def get_issue_types():
    """Get available issue types"""
    from advanced_bigshorts_agent import ALLOWED_ISSUE_TYPES
    return {
        "issue_types": ALLOWED_ISSUE_TYPES,
        "total": len(ALLOWED_ISSUE_TYPES)
    }

# WebSocket for streaming (optional)
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            # Process with agent
            response = agent.process_query(message, session_id)
            
            # Send response
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session: {session_id}")

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
