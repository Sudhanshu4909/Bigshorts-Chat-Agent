# Advanced BigShorts Agent with RAG and Multi-Session Support
import os
import json
from typing import Dict, List, Union, Optional, Any
from datetime import datetime
from pathlib import Path

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pydantic import BaseModel, Field

# Optional: Vector store for RAG
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG features unavailable. Install chromadb or faiss-cpu for RAG support.")

# Import constants from the basic version
from bigshorts_langchain_agent import (
    ALLOWED_CONTENT_TYPES, ALLOWED_ISSUE_TYPES, CONTENT_TYPE_MAPPING,
    CONTENT_GUIDES, ISSUE_SOLUTIONS,
    content_creation_tool_func, handle_issue_tool_func,
    platform_guide_tool_func, generate_interactive_ideas_func,
    get_trending_content_func
)


class AdvancedBigShortsAgent:
    """Advanced agent with RAG, session management, and analytics"""
    
    def __init__(self, model_path: str, enable_rag: bool = True):
        """Initialize the advanced agent"""
        
        self.model_path = model_path
        self.enable_rag = enable_rag and RAG_AVAILABLE
        self.sessions: Dict[str, Dict] = {}
        
        # Initialize LLM
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4096,  # Larger context for RAG
            n_gpu_layers=-1,  # Use GPU if available
            n_threads=8,
            n_batch=512,
            temperature=0.7,
            max_tokens=512,
            top_p=0.95,
            callback_manager=callback_manager,
            verbose=False,
        )
        
        # Initialize RAG if enabled
        if self.enable_rag:
            self._setup_rag()
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent template
        self.prompt_template = self._create_prompt_template()
        
        # Analytics
        self.analytics = {
            "total_queries": 0,
            "tool_usage": {},
            "popular_content_types": {},
            "common_issues": {}
        }
    
    def _setup_rag(self):
        """Setup RAG with vector store"""
        try:
            # Create embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create knowledge base documents
            documents = []
            
            # Add content guides to knowledge base
            for content_type, guide in CONTENT_GUIDES.items():
                doc_text = f"Content Type: {content_type}\n"
                doc_text += f"Title: {guide['title']}\n\n"
                for step in guide.get('steps', []):
                    doc_text += f"Step {step['step']}: {step['description']}\n"
                    if 'tips' in step:
                        doc_text += f"Tip: {step['tips']}\n"
                
                documents.append(Document(
                    page_content=doc_text,
                    metadata={"type": "content_guide", "content_type": content_type}
                ))
            
            # Add issue solutions to knowledge base
            for issue_type, solution in ISSUE_SOLUTIONS.items():
                documents.append(Document(
                    page_content=f"Issue: {issue_type}\nSolution: {solution}",
                    metadata={"type": "issue_solution", "issue_type": issue_type}
                ))
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            print("RAG enabled: Knowledge base created with vector store")
            
        except Exception as e:
            print(f"RAG setup failed: {e}")
            self.enable_rag = False
    
    def _rag_search_tool(self, query: str) -> str:
        """Search the knowledge base using RAG"""
        if not self.enable_rag:
            return "RAG not available"
        
        try:
            # Search for relevant documents
            docs = self.vectorstore.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant information found"
            
            # Format results
            results = "Here's what I found in the knowledge base:\n\n"
            for i, doc in enumerate(docs, 1):
                results += f"{i}. {doc.page_content[:200]}...\n\n"
            
            return results
            
        except Exception as e:
            return f"Error searching knowledge base: {e}"
    
    def _create_tools(self) -> List[Tool]:
        """Create all available tools"""
        tools = [
            StructuredTool(
                name="content_creation_guide",
                description="Provides step-by-step guides for creating SHOT, SNIP, SSUP, Collab, or Mini content",
                func=content_creation_tool_func,
                args_schema=type('ContentGuideInput', (BaseModel,), {
                    '__annotations__': {'content_type': str},
                    'content_type': Field(description="Type of content (shot, snip, ssup, collab, Mini)")
                })
            ),
            StructuredTool(
                name="handle_issue",
                description="Solves platform issues like login, upload, notification problems",
                func=handle_issue_tool_func,
                args_schema=type('IssueInput', (BaseModel,), {
                    '__annotations__': {'issue_type': str},
                    'issue_type': Field(description="Type of issue (login, upload, notification, etc.)")
                })
            ),
            Tool(
                name="platform_guide",
                description="Explains platform features and sections",
                func=platform_guide_tool_func
            ),
            Tool(
                name="interactive_ideas",
                description="Generates creative ideas for interactive Snip videos",
                func=generate_interactive_ideas_func
            ),
            Tool(
                name="trending_content",
                description="Shows trending content and popular creators",
                func=get_trending_content_func
            ),
        ]
        
        # Add RAG tool if available
        if self.enable_rag:
            tools.append(
                Tool(
                    name="knowledge_search",
                    description="Search the BigShorts knowledge base for detailed information",
                    func=self._rag_search_tool
                )
            )
        
        return tools
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the agent prompt template"""
        template = """You are Gyan.Ai, the intelligent assistant for BigShorts social media platform.

CORE RESPONSIBILITIES:
- Help users create content: SHOT (photos), SNIP (short videos), SSUP (stories), Mini (long videos), Collab (collaborative posts)
- Troubleshoot platform issues
- Explain features and provide guidance
- Suggest trending content and creative ideas

IMPORTANT GUIDELINES:
1. Stay focused on BigShorts features only
2. Be concise, friendly, and helpful
3. Use tools to provide detailed information
4. For off-topic questions, politely redirect to BigShorts
5. Provide step-by-step guidance when needed

Available tools: {tools}
Tool names: {tool_names}

RESPONSE FORMAT:
Question: the input question you must answer
Thought: consider what information is needed
Action: choose the best tool from [{tool_names}]
Action Input: provide the required input for the tool
Observation: analyze the tool's output
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: formulate the final answer
Final Answer: provide a clear, helpful response

Conversation History:
{chat_history}

Current Question: {input}
{agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
    
    def create_session(self, session_id: str) -> None:
        """Create a new conversation session"""
        if session_id not in self.sessions:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt_template
            )
            
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate"
            )
            
            self.sessions[session_id] = {
                "executor": executor,
                "memory": memory,
                "created_at": datetime.now().isoformat(),
                "query_count": 0,
                "last_activity": datetime.now().isoformat()
            }
    
    def process_query(self, user_input: str, session_id: str = "default") -> Union[str, dict]:
        """Process a user query with session support"""
        
        # Create session if it doesn't exist
        self.create_session(session_id)
        
        # Update analytics
        self.analytics["total_queries"] += 1
        self.sessions[session_id]["query_count"] += 1
        self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        
        # Handle greetings
        greetings = ["hello", "hi", "hey", "greetings", "howdy", "sup", "yo"]
        if user_input.lower().strip() in greetings:
            faqs = [
                {"question": "How to create a MINI?", "content_type": "Mini", "query": "How to create a Mini"},
                {"question": "Promote Your MINI on a SNIP", "content_type": "Snip to Mini", "query": "How to link Snip to Mini"},
                {"question": "How to create Interactive Content?", "content_type": "Interactive snip", "query": "How to create Interactive Snip"},
                {"question": "How do I create a SHOT?", "content_type": "shot", "query": "How to create a shot"},
                {"question": "How do I create a SNIP?", "content_type": "snip", "query": "How to create a snip"},
            ]
            
            return {
                "type": "greeting_with_faqs",
                "content": {
                    "greeting": "Hello! 😊 I'm Gyan.Ai, your BigShorts assistant. What would you like to create today?",
                    "faqs": faqs
                }
            }
        
        # Handle FAQ selections
        if user_input.startswith("FAQ:"):
            content_type = user_input.split("FAQ:")[1].strip()
            
            # Track popular content types
            if content_type in self.analytics["popular_content_types"]:
                self.analytics["popular_content_types"][content_type] += 1
            else:
                self.analytics["popular_content_types"][content_type] = 1
            
            return content_creation_tool_func(content_type)
        
        # Process through agent
        try:
            session = self.sessions[session_id]
            result = session["executor"].invoke({"input": user_input})
            
            output = result.get("output", "I couldn't process that request.")
            
            if isinstance(output, dict):
                return output
            
            return {"type": "message", "content": output}
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "type": "error",
                "content": "I encountered an issue. Can I help you with creating SHOT, SNIP, SSUP, Mini, or Collab content?"
            }
    
    def get_session_history(self, session_id: str = "default") -> List[str]:
        """Get conversation history for a session"""
        if session_id in self.sessions:
            memory = self.sessions[session_id]["memory"]
            return [msg.content for msg in memory.chat_memory.messages]
        return []
    
    def get_analytics(self) -> Dict:
        """Get usage analytics"""
        return {
            **self.analytics,
            "active_sessions": len(self.sessions),
            "sessions": {
                sid: {
                    "query_count": sdata["query_count"],
                    "created_at": sdata["created_at"],
                    "last_activity": sdata["last_activity"]
                }
                for sid, sdata in self.sessions.items()
            }
        }
    
    def save_analytics(self, filepath: str = "analytics.json"):
        """Save analytics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_analytics(), f, indent=2)
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Demo function
def run_advanced_agent():
    """Run the advanced agent with multi-session support"""
    model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Initializing Advanced BigShorts Agent...")
    agent = AdvancedBigShortsAgent(model_path, enable_rag=True)
    print("\nAdvanced Agent ready! Features: Multi-session, RAG, Analytics")
    print("Commands: 'exit' to quit, 'analytics' to view stats, 'new session' to start fresh\n")
    
    current_session = "default"
    
    while True:
        user_input = input(f"\n[Session: {current_session}] You: ").strip()
        
        if user_input.lower() == "exit":
            print("\nAgent: Thanks for using BigShorts! Have a great day!")
            agent.save_analytics()
            break
        
        if user_input.lower() == "analytics":
            print("\n--- Analytics ---")
            print(json.dumps(agent.get_analytics(), indent=2))
            continue
        
        if user_input.lower() == "new session":
            import random
            current_session = f"session_{random.randint(1000, 9999)}"
            print(f"\nStarted new session: {current_session}")
            continue
        
        if not user_input:
            continue
        
        # Process query
        response = agent.process_query(user_input, session_id=current_session)
        
        # Display response
        if isinstance(response, dict):
            if response.get("type") == "message":
                print(f"\nAgent: {response.get('content')}")
            elif response.get("type") == "content_guide":
                guide = response.get("content", {})
                print(f"\nAgent: Here's the guide for {guide.get('title')}!")
                steps = guide.get('steps', [])
                for step in steps[:3]:  # Show first 3 steps
                    print(f"  Step {step['step']}: {step['description']}")
                if len(steps) > 3:
                    print(f"  ... and {len(steps) - 3} more steps")
            elif response.get("type") == "greeting_with_faqs":
                print(f"\nAgent: {response['content']['greeting']}")
                print("\nPopular questions:")
                for faq in response['content']['faqs'][:5]:
                    print(f"  - {faq['question']}")
            else:
                print(f"\nAgent: {response}")
        else:
            print(f"\nAgent: {response}")


if __name__ == "__main__":
    run_advanced_agent()
