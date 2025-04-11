from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, Dict, List, Any, Optional
import os
import uvicorn
from Chatbot2 import BigshortsChatbot
import asyncio
import traceback
import json
import uuid
import threading
from datetime import datetime, timedelta

app = FastAPI(title="Bigshorts Chatbot API")

# Enable CORS for all domains on all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://bigshortsbot-fba2hdbpajcfa9d4.centralindia-01.azurewebsites.net",
        "http://20.197.5.250",
        "http://52.140.106.225" # Be cautious with this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model not found at {MODEL_PATH}")
    print("The API will start but chatbot functionality won't work until the model is available")

# Dictionary to store chatbot instances for each session
chatbot_instances = {}
chatbot_lock = threading.Lock()

# Last access time for each session
last_access = {}

# Max number of chatbot instances to keep in memory
MAX_CHATBOTS = 100

# Session timeout in minutes
SESSION_TIMEOUT = 30

# Request model
class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None

# FAQ selection model
class FAQSelectRequest(BaseModel):
    content_type: str
    session_id: Optional[str] = None

# Simplified response model without validation
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """API endpoint to process chat messages"""
    try:
        if not request.content:
            return {"type": "error", "content": "No message provided", "session_id": request.session_id or str(uuid.uuid4())}
        
        # Use provided session ID or generate a new one
        session_id = request.session_id or str(uuid.uuid4())
        
        # Clean up old sessions occasionally
        clean_old_sessions()
        
        # Get or create chatbot for this session
        chatbot = get_session_chatbot(session_id)
        if chatbot is None:
            return {"type": "error", "content": "Failed to initialize chatbot", "session_id": session_id}
        
        try:
            # Process the request
            print(f"Processing query for session {session_id}: {request.content}")
            response = chatbot.process_query(request.content)
            print(f"Raw chatbot response: {type(response)} - {response}")
            
            # Update last access time
            with chatbot_lock:
                last_access[session_id] = datetime.now()
            
            # Handle different response types
            if response is None:
                print("WARNING: Chatbot returned None response")
                return {"type": "message", "content": "I'm sorry, I couldn't process that request.", "session_id": session_id}
            
            if isinstance(response, dict):
                if "content" in response and response["content"] is None:
                    print("WARNING: Response has None content")
                    response["content"] = "I'm sorry, I encountered an issue processing that request."
                
                if "type" not in response:
                    print("WARNING: Response missing type field")
                    response["type"] = "message"
                
                # Add session_id to the response
                response["session_id"] = session_id
                return response
            else:
                # For string or other responses
                return {"type": "message", "content": str(response), "session_id": session_id}
                
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            traceback.print_exc()
            return {"type": "error", "content": f"Processing error: {str(e)}", "session_id": session_id}
    
    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return {"type": "error", "content": f"Server error: {str(e)}", "session_id": request.session_id or str(uuid.uuid4())}

@app.post("/api/select-faq")
async def select_faq(request: FAQSelectRequest):
    """API endpoint to handle FAQ selection"""
    try:
        # Use provided session ID or generate a new one
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create chatbot for this session
        chatbot = get_session_chatbot(session_id)
        if chatbot is None:
            return {"type": "error", "content": "Failed to initialize chatbot", "session_id": session_id}
        
        # Format the request as if the user had asked about this topic
        formatted_request = f"FAQ: {request.content_type}"
        
        # Process the request
        response = chatbot.process_query(formatted_request)
        
        # Update last access time
        with chatbot_lock:
            last_access[session_id] = datetime.now()
        
        # Add session_id to the response
        if isinstance(response, dict):
            response["session_id"] = session_id
        else:
            response = {"type": "message", "content": str(response), "session_id": session_id}
            
        return response
        
    except Exception as e:
        print(f"Error selecting FAQ: {str(e)}")
        traceback.print_exc()
        return {"type": "error", "content": f"Error processing FAQ selection: {str(e)}", "session_id": request.session_id or str(uuid.uuid4())}

def get_session_chatbot(session_id):
    """Get or create a chatbot instance for a session"""
    with chatbot_lock:
        # If chatbot already exists for this session, return it
        if session_id in chatbot_instances:
            return chatbot_instances[session_id]
        
        # If we have too many chatbots, remove the oldest one
        if len(chatbot_instances) >= MAX_CHATBOTS:
            oldest_session = min(last_access.items(), key=lambda x: x[1])[0]
            if oldest_session in chatbot_instances:
                del chatbot_instances[oldest_session]
                del last_access[oldest_session]
        
        # Create a new chatbot for this session
        try:
            chatbot = BigshortsChatbot(MODEL_PATH)
            chatbot_instances[session_id] = chatbot
            last_access[session_id] = datetime.now()
            return chatbot
        except Exception as e:
            print(f"Failed to initialize chatbot for session {session_id}: {str(e)}")
            traceback.print_exc()
            return None

def clean_old_sessions():
    """Remove chatbot instances for sessions that haven't been accessed in a while"""
    with chatbot_lock:
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, access_time in last_access.items():
            if current_time - access_time > timedelta(minutes=SESSION_TIMEOUT):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            if session_id in chatbot_instances:
                del chatbot_instances[session_id]
                del last_access[session_id]
                print(f"Removed inactive session: {session_id}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "ok", 
        "model_loaded": model_loaded, 
        "active_sessions": len(chatbot_instances)
    }

@app.get("/api/sessions")
async def get_sessions():
    """Get information about active sessions"""
    with chatbot_lock:
        sessions = []
        for session_id, access_time in last_access.items():
            idle_time = (datetime.now() - access_time).total_seconds() / 60
            sessions.append({
                "session_id": session_id,
                "idle_minutes": round(idle_time, 2)
            })
        
        return {"active_sessions": len(chatbot_instances), "sessions": sessions}

@app.post("/api/clear-session")
async def clear_session(session_id: str):
    """Clear a specific session"""
    with chatbot_lock:
        if session_id in chatbot_instances:
            del chatbot_instances[session_id]
            if session_id in last_access:
                del last_access[session_id]
            return {"status": "success", "message": "Session cleared"}
    
    return {"status": "warning", "message": "Session not found"}

if __name__ == "__main__":
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=5000)