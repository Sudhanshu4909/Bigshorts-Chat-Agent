from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, Dict, List, Any, Optional
import os
import uvicorn
from Chatbot2 import BigShortsChatbot
import asyncio
import traceback
import json
import uuid
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Bigshorts Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://bigshortsbot-fba2hdbpajcfa9d4.centralindia-01.azurewebsites.net",
        "http://20.197.5.250",
        "http://52.140.106.225"
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

# OPTIMIZED FOR 8 vCPUs, 128 GiB RAM
# Shared chatbot instance
chatbot_instance = None
chatbot_lock = threading.Lock()

# Thread pool for CPU-intensive operations
# Use 6 threads (leaving 2 CPUs for system/async tasks)
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="chatbot_worker")

# Request Queue Configuration - Aggressive settings for powerful hardware
MAX_QUEUE_SIZE = 500  # Large queue to handle traffic spikes
MAX_CONCURRENT_REQUESTS = 20  # Higher concurrency with 8 vCPUs
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
request_queue_size = 0
queue_lock = threading.Lock()

# Rate Limiting Configuration - More permissive for better UX
RATE_LIMIT_REQUESTS = 30  # 30 requests per window (up from 10)
RATE_LIMIT_WINDOW = 60  # 60 seconds
rate_limit_data = defaultdict(lambda: deque())  # session_id -> deque of timestamps

# Session management
last_access = {}
SESSION_TIMEOUT = 60  # Increased to 60 minutes with more RAM

# Stats tracking
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "rate_limited_requests": 0,
    "queue_full_requests": 0,
    "average_response_time": 0.0,
    "response_times": deque(maxlen=1000)  # Keep last 1000 response times
}
stats_lock = threading.Lock()

def get_chatbot():
    """Get the shared chatbot instance (lazy loading) - optimized for high RAM"""
    global chatbot_instance
    
    if chatbot_instance is None:
        with chatbot_lock:
            # Double-check after acquiring lock
            if chatbot_instance is None:
                print("Initializing shared chatbot instance with optimized settings...")
                try:
                    # Initialize with settings optimized for your hardware
                    from llama_cpp import Llama
                    
                    # Create a custom initialized chatbot
                    chatbot_instance = BigShortsChatbot.__new__(BigShortsChatbot)
                    
                    # Initialize the LLM with optimized settings for 8 vCPUs
                    chatbot_instance.llm = Llama(
                        model_path=MODEL_PATH,
                        n_ctx=4096,  # Larger context with more RAM
                        n_gpu_layers=0,  # CPU only
                        n_threads=6,  # Use 6 threads
                        n_batch=512,  # Larger batch size
                        use_mlock=True,  # Lock model in RAM (you have 128GB!)
                        use_mmap=True,  # Memory map for efficiency
                        verbose=False
                    )
                    
                    # Initialize other attributes
                    import yaml
                    try:
                        with open("prompts.yaml", 'r') as stream:
                            chatbot_instance.prompt_templates = yaml.safe_load(stream)
                    except:
                        chatbot_instance.prompt_templates = {
                            "final_answer": {
                                "pre_messages": "You are a helpful social media assistant for the BigShorts platform.",
                                "post_messages": "Remember to never show your reasoning or thought process to the user."
                            }
                        }
                    
                    chatbot_instance.sessions = {}
                    chatbot_instance.off_topic_keywords = [
                        "politics", "news", "weather", "sports", "dating", "games", "gaming"
                    ]
                    chatbot_instance.unsupported_query_response = {
                        "type": "error",
                        "content": "I can only help with BigShorts platform features."
                    }
                    chatbot_instance.content_explanations = {}
                    
                    print("Chatbot initialized successfully with optimized settings!")
                except Exception as e:
                    print(f"Error with custom initialization, falling back to default: {e}")
                    chatbot_instance = BigShortsChatbot(MODEL_PATH)
    
    return chatbot_instance

def update_stats(response_time: float, success: bool):
    """Update request statistics"""
    with stats_lock:
        request_stats["total_requests"] += 1
        if success:
            request_stats["successful_requests"] += 1
        else:
            request_stats["failed_requests"] += 1
        
        request_stats["response_times"].append(response_time)
        
        # Calculate rolling average
        if request_stats["response_times"]:
            request_stats["average_response_time"] = sum(request_stats["response_times"]) / len(request_stats["response_times"])

def check_rate_limit(session_id: str) -> tuple[bool, int]:
    """
    Check if the session has exceeded rate limits
    Returns: (is_allowed, remaining_requests)
    """
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW
    
    # Get request timestamps for this session
    request_times = rate_limit_data[session_id]
    
    # Remove timestamps outside the current window
    while request_times and request_times[0] < window_start:
        request_times.popleft()
    
    # Check if limit exceeded
    requests_in_window = len(request_times)
    remaining = RATE_LIMIT_REQUESTS - requests_in_window
    
    if requests_in_window >= RATE_LIMIT_REQUESTS:
        with stats_lock:
            request_stats["rate_limited_requests"] += 1
        return False, 0
    
    # Add current request timestamp
    request_times.append(current_time)
    
    return True, remaining

def check_queue_capacity() -> tuple[bool, int]:
    """
    Check if the queue has capacity
    Returns: (has_capacity, current_queue_size)
    """
    global request_queue_size
    with queue_lock:
        if request_queue_size >= MAX_QUEUE_SIZE:
            with stats_lock:
                request_stats["queue_full_requests"] += 1
            return False, request_queue_size
        request_queue_size += 1
        return True, request_queue_size

def release_queue_slot():
    """Release a slot in the queue"""
    global request_queue_size
    with queue_lock:
        request_queue_size = max(0, request_queue_size - 1)

# Request model
class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None

# FAQ selection model
class FAQSelectRequest(BaseModel):
    content_type: str
    session_id: Optional[str] = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """API endpoint to process chat messages with rate limiting and queuing"""
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        if not request.content:
            return {
                "type": "error", 
                "content": "No message provided", 
                "session_id": session_id
            }
        
        # Check rate limit
        is_allowed, remaining = check_rate_limit(session_id)
        if not is_allowed:
            return {
                "type": "error",
                "content": f"Rate limit exceeded. You can make {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
                "session_id": session_id,
                "rate_limit_exceeded": True,
                "retry_after": RATE_LIMIT_WINDOW
            }
        
        # Check queue capacity
        has_capacity, queue_size = check_queue_capacity()
        if not has_capacity:
            return {
                "type": "error",
                "content": "Server is at capacity. Please try again in a moment.",
                "session_id": session_id,
                "queue_full": True,
                "queue_size": queue_size
            }
        
        try:
            # Acquire semaphore to limit concurrent processing
            async with request_semaphore:
                # Get the shared chatbot instance
                chatbot = get_chatbot()
                if chatbot is None:
                    update_stats(time.time() - start_time, False)
                    return {
                        "type": "error", 
                        "content": "Failed to initialize chatbot", 
                        "session_id": session_id
                    }
                
                # Process the request
                print(f"[Session: {session_id[:8]}...] Processing: {request.content[:50]}...")
                print(f"Queue: {queue_size}/{MAX_QUEUE_SIZE}, Rate limit remaining: {remaining}/{RATE_LIMIT_REQUESTS}")
                
                # Run the blocking chatbot.process_query in thread pool executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    executor,
                    chatbot.process_query, 
                    request.content, 
                    session_id
                )
                
                # Update last access time
                with chatbot_lock:
                    last_access[session_id] = datetime.now()
                
                # Calculate response time
                response_time = time.time() - start_time
                update_stats(response_time, True)
                
                print(f"[Session: {session_id[:8]}...] Completed in {response_time:.2f}s")
                
                # Handle different response types
                if response is None:
                    print("WARNING: Chatbot returned None response")
                    return {
                        "type": "message", 
                        "content": "I'm sorry, I couldn't process that request.", 
                        "session_id": session_id,
                        "response_time": response_time
                    }
                
                if isinstance(response, dict):
                    if "content" in response and response["content"] is None:
                        print("WARNING: Response has None content")
                        response["content"] = "I'm sorry, I encountered an issue processing that request."
                    
                    if "type" not in response:
                        print("WARNING: Response missing type field")
                        response["type"] = "message"
                    
                    # Add metadata to response
                    response["session_id"] = session_id
                    response["rate_limit_remaining"] = remaining
                    response["response_time"] = round(response_time, 2)
                    return response
                else:
                    # For string or other responses
                    return {
                        "type": "message", 
                        "content": str(response), 
                        "session_id": session_id,
                        "rate_limit_remaining": remaining,
                        "response_time": round(response_time, 2)
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            update_stats(response_time, False)
            print(f"Error processing message: {str(e)}")
            traceback.print_exc()
            return {
                "type": "error", 
                "content": f"Processing error: {str(e)}", 
                "session_id": session_id
            }
        finally:
            # Always release the queue slot
            release_queue_slot()
    
    except Exception as e:
        response_time = time.time() - start_time
        update_stats(response_time, False)
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        release_queue_slot()
        return {
            "type": "error", 
            "content": f"Server error: {str(e)}", 
            "session_id": session_id
        }

@app.post("/api/select-faq")
async def select_faq(request: FAQSelectRequest):
    """API endpoint to handle FAQ selection with rate limiting and queuing"""
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Check rate limit
        is_allowed, remaining = check_rate_limit(session_id)
        if not is_allowed:
            return {
                "type": "error",
                "content": f"Rate limit exceeded. You can make {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
                "session_id": session_id,
                "rate_limit_exceeded": True,
                "retry_after": RATE_LIMIT_WINDOW
            }
        
        # Check queue capacity
        has_capacity, queue_size = check_queue_capacity()
        if not has_capacity:
            return {
                "type": "error",
                "content": "Server is at capacity. Please try again in a moment.",
                "session_id": session_id,
                "queue_full": True,
                "queue_size": queue_size
            }
        
        try:
            # Acquire semaphore to limit concurrent processing
            async with request_semaphore:
                # Get the shared chatbot instance
                chatbot = get_chatbot()
                if chatbot is None:
                    update_stats(time.time() - start_time, False)
                    return {
                        "type": "error", 
                        "content": "Failed to initialize chatbot", 
                        "session_id": session_id
                    }
                
                # Format the request
                formatted_request = f"FAQ: {request.content_type}"
                
                # Process the request in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    executor,
                    chatbot.process_query,
                    formatted_request,
                    session_id
                )
                
                # Update last access time
                with chatbot_lock:
                    last_access[session_id] = datetime.now()
                
                response_time = time.time() - start_time
                update_stats(response_time, True)
                
                # Add metadata to response
                if isinstance(response, dict):
                    response["session_id"] = session_id
                    response["rate_limit_remaining"] = remaining
                    response["response_time"] = round(response_time, 2)
                else:
                    response = {
                        "type": "message", 
                        "content": str(response), 
                        "session_id": session_id,
                        "rate_limit_remaining": remaining,
                        "response_time": round(response_time, 2)
                    }
                    
                return response
        finally:
            release_queue_slot()
        
    except Exception as e:
        response_time = time.time() - start_time
        update_stats(response_time, False)
        print(f"Error selecting FAQ: {str(e)}")
        traceback.print_exc()
        release_queue_slot()
        return {
            "type": "error", 
            "content": f"Error processing FAQ selection: {str(e)}", 
            "session_id": session_id
        }

def clean_old_sessions():
    """Remove conversation history for sessions that haven't been accessed in a while"""
    chatbot = get_chatbot()
    if chatbot is None:
        return
        
    with chatbot_lock:
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, access_time in list(last_access.items()):
            if current_time - access_time > timedelta(minutes=SESSION_TIMEOUT):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            # Remove conversation history from chatbot
            if session_id in chatbot.sessions:
                del chatbot.sessions[session_id]
            if session_id in last_access:
                del last_access[session_id]
            # Clean up rate limit data
            if session_id in rate_limit_data:
                del rate_limit_data[session_id]
            print(f"Cleaned up inactive session: {session_id[:8]}...")

@app.get("/api/health")
async def health_check():
    """Health check endpoint with detailed stats"""
    model_loaded = os.path.exists(MODEL_PATH)
    chatbot = get_chatbot() if model_loaded else None
    
    with queue_lock:
        current_queue_size = request_queue_size
    
    with stats_lock:
        stats_copy = request_stats.copy()
        stats_copy["response_times"] = list(stats_copy["response_times"])[-10:]  # Last 10
    
    # Calculate success rate
    total = stats_copy["total_requests"]
    success_rate = (stats_copy["successful_requests"] / total * 100) if total > 0 else 0
    
    return {
        "status": "ok", 
        "model_loaded": model_loaded,
        "chatbot_initialized": chatbot is not None,
        "active_sessions": len(last_access),
        "queue_size": current_queue_size,
        "max_queue_size": MAX_QUEUE_SIZE,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "rate_limit_requests": RATE_LIMIT_REQUESTS,
        "rate_limit_window": RATE_LIMIT_WINDOW,
        "session_timeout_minutes": SESSION_TIMEOUT,
        "statistics": {
            "total_requests": stats_copy["total_requests"],
            "successful_requests": stats_copy["successful_requests"],
            "failed_requests": stats_copy["failed_requests"],
            "rate_limited_requests": stats_copy["rate_limited_requests"],
            "queue_full_requests": stats_copy["queue_full_requests"],
            "success_rate_percent": round(success_rate, 2),
            "average_response_time_seconds": round(stats_copy["average_response_time"], 2),
            "recent_response_times": stats_copy["response_times"]
        },
        "hardware": {
            "vcpus": 8,
            "ram_gb": 128,
            "worker_threads": 6,
            "model_context_size": 4096
        }
    }

@app.get("/api/sessions")
async def get_sessions():
    """Get information about active sessions"""
    chatbot = get_chatbot()
    with chatbot_lock:
        sessions = []
        for session_id, access_time in last_access.items():
            idle_time = (datetime.now() - access_time).total_seconds() / 60
            
            # Get rate limit info
            current_time = time.time()
            window_start = current_time - RATE_LIMIT_WINDOW
            request_times = rate_limit_data.get(session_id, deque())
            recent_requests = sum(1 for t in request_times if t >= window_start)
            
            sessions.append({
                "session_id": session_id[:8] + "...",  # Truncate for privacy
                "idle_minutes": round(idle_time, 2),
                "recent_requests": recent_requests,
                "rate_limit_remaining": RATE_LIMIT_REQUESTS - recent_requests,
                "conversation_length": len(chatbot.sessions.get(session_id, []))
            })
        
        with queue_lock:
            current_queue_size = request_queue_size
        
        return {
            "active_sessions": len(last_access),
            "sessions": sessions,
            "total_conversation_sessions": len(chatbot.sessions) if chatbot else 0,
            "current_queue_size": current_queue_size,
            "active_rate_limited_sessions": len(rate_limit_data)
        }

@app.post("/api/clear-session")
async def clear_session(session_id: str):
    """Clear a specific session's conversation history"""
    chatbot = get_chatbot()
    if chatbot is None:
        return {"status": "error", "message": "Chatbot not initialized"}
        
    with chatbot_lock:
        cleared_items = []
        if session_id in chatbot.sessions:
            del chatbot.sessions[session_id]
            cleared_items.append("conversation_history")
        if session_id in last_access:
            del last_access[session_id]
            cleared_items.append("access_time")
        if session_id in rate_limit_data:
            del rate_limit_data[session_id]
            cleared_items.append("rate_limit_data")
        
        if cleared_items:
            return {"status": "success", "message": f"Session cleared: {', '.join(cleared_items)}"}
    
    return {"status": "warning", "message": "Session not found"}

@app.get("/api/rate-limit/{session_id}")
async def get_rate_limit_status(session_id: str):
    """Get rate limit status for a specific session"""
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW
    request_times = rate_limit_data.get(session_id, deque())
    
    # Count requests in current window
    recent_requests = sum(1 for t in request_times if t >= window_start)
    remaining = RATE_LIMIT_REQUESTS - recent_requests
    
    # Calculate reset time
    if request_times:
        oldest_in_window = min([t for t in request_times if t >= window_start], default=current_time)
        reset_in = max(0, int(RATE_LIMIT_WINDOW - (current_time - oldest_in_window)))
    else:
        reset_in = 0
    
    return {
        "session_id": session_id[:8] + "...",
        "requests_in_window": recent_requests,
        "remaining": remaining,
        "limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
        "reset_in_seconds": reset_in,
        "is_rate_limited": remaining <= 0
    }

@app.get("/api/stats")
async def get_stats():
    """Get detailed server statistics"""
    with stats_lock:
        stats_copy = request_stats.copy()
        response_times_list = list(stats_copy["response_times"])
    
    total = stats_copy["total_requests"]
    success_rate = (stats_copy["successful_requests"] / total * 100) if total > 0 else 0
    
    # Calculate percentiles
    if response_times_list:
        sorted_times = sorted(response_times_list)
        p50 = sorted_times[int(len(sorted_times) * 0.5)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
    else:
        p50 = p95 = p99 = 0
    
    return {
        "total_requests": total,
        "successful_requests": stats_copy["successful_requests"],
        "failed_requests": stats_copy["failed_requests"],
        "rate_limited_requests": stats_copy["rate_limited_requests"],
        "queue_full_requests": stats_copy["queue_full_requests"],
        "success_rate_percent": round(success_rate, 2),
        "response_times": {
            "average": round(stats_copy["average_response_time"], 2),
            "p50_median": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(min(response_times_list), 2) if response_times_list else 0,
            "max": round(max(response_times_list), 2) if response_times_list else 0
        }
    }

# Background task to clean old sessions periodically
@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup"""
    print(f"Starting BigShorts Chatbot API")
    print(f"Hardware: 8 vCPUs, 128 GiB RAM")
    print(f"Configuration: {MAX_CONCURRENT_REQUESTS} concurrent, {MAX_QUEUE_SIZE} queue size")
    print(f"Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")
    
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            try:
                clean_old_sessions()
                print(f"Periodic cleanup: {len(last_access)} active sessions")
            except Exception as e:
                print(f"Error in periodic cleanup: {e}")
    
    asyncio.create_task(periodic_cleanup())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down server...")
    executor.shutdown(wait=True)
    print("Executor shutdown complete")

if __name__ == "__main__":
    # Start the server with optimized settings
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5000,
        workers=1,  # Single worker since we handle concurrency internally
        limit_concurrency=MAX_CONCURRENT_REQUESTS * 2,
        timeout_keep_alive=75
    )