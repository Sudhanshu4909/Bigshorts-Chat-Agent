# BigShorts Autonomous Agent using LangChain
import os
import random
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field

# Configuration constants
ALLOWED_CONTENT_TYPES = [
    "shot", "snip", "ssup", "collab",
    "editing a shot", "invite friends", "feedback", "multiple accounts", 
    "account overview", "store draft", "change password", "notification", 
    "change theme", "report", "moment", "delete post", "post insights", 
    "saved posts", "edit profile", "edit post", "block/unblock user", 
    "hide/unhide users", "messages", "discovery", "editing a ssup", 
    "interactive snip", "Mini", "create a playlist", "editing a Mini", 
    "editing a snip", "Mini Series", "Edit Mini Series", "Ssup repost",
    "Snip to Mini", "bigcoins_reward"
]

ALLOWED_ISSUE_TYPES = [
    "login", "upload", "notification", "privacy", 
    "account", "payment", "content", "technical", "app", 
    "video", "audio", "connection", "quality", "blocking", 
    "reporting", "messaging", "password", "theme"
]

# Content type mapping (abbreviated for brevity - use full mapping from original)
CONTENT_TYPE_MAPPING = {
    "photo": "shot", "picture": "shot", "image": "shot",
    "video": "snip", "clip": "snip", "reel": "snip",
    "story": "ssup", "stories": "ssup",
    "collaboration": "collab", "together": "collab",
    "edit shot": "editing a shot", "edit snip": "editing a snip",
    "long video": "Mini", "Mini video": "Mini",
    # Add all mappings from original file...
}

# Pydantic models for structured tool inputs
class ContentGuideInput(BaseModel):
    content_type: str = Field(description="The type of content to create (e.g., 'shot', 'snip', 'ssup', 'collab', 'Mini')")

class IssueInput(BaseModel):
    issue_type: str = Field(description="The type of issue to resolve (e.g., 'login', 'upload', 'notification')")

class PlatformGuideInput(BaseModel):
    section: str = Field(description="The platform section to get guidance about")

# Data storage for guides
CONTENT_GUIDES = {
    "shot": {
        "title": "Creating a BigShorts SHOT",
        "steps": [
            {
                "step": 1,
                "description": "Open the BigShorts app and tap the Creation Button",
                "image_path": "Shot/Group_1449.webp",
                "tips": "Make sure you're on the latest app version for all features"
            },
            {
                "step": 2,
                "description": "Choose 'SHOT' from the Creation Wheel",
                "image_path": "Shot/Group_1450.webp",
            },
            {
                "step": 3,
                "description": "Capture a SHOT or upload an existing photo from your Device",
                "image_path": "Shot/Group_1451.webp",
                "tips": "SHOT can include multiple Pictures"
            },
            {
                "step": 4,
                "description": "Edit your SHOT using BigShorts tools",
                "image_path": "Shot/Group_1452.webp",
                "tips": "Try our AI-powered filters and effects"
            },
            {
                "step": 5,
                "description": "Add captions, hashtags, and description Or Collab with your Friends and post",
                "image_path": "Shot/Group_1453.webp",
                "tips": "Use trending hashtags for better reach"
            }
        ],
    },
    "snip": {
        "title": "Creating a BigShorts SNIP",
        "steps": [
            {
                "step": 1,
                "description": "Open the BigShorts app and tap the Creation Button",
                "image_path": "Shot/Group_1444.webp",
                "tips": "Ensure stable internet connection"
            },
            {
                "step": 2,
                "description": "Choose 'SNIP' from the Creation Wheel",
                "image_path": "Shot/Group_1445.webp",
            },
            {
                "step": 3,
                "description": "Capture video or choose a video and click next",
                "image_path": "Shot/Group_1523.webp",
            },
            {
                "step": 4,
                "description": "Edit your SNIP using BigShorts tools and tap done",
                "image_path": "Shot/Group_1447.webp",
            },
            {
                "step": 5,
                "description": "Add captions, hashtags, and description Or Collab with your Friends",
                "image_path": "Shot/Group_1448.webp",
                "tips": "Use trending hashtags for better reach"
            }
        ],
    },
    # Add other content types...
}

ISSUE_SOLUTIONS = {
    "login": "If you're having trouble logging in:\n1. Check your username/password\n2. Clear Application cache\n3. Reset password if needed",
    "upload": "For upload issues:\n1. Check file size (max 20MB)\n2. Ensure supported format\n3. Check internet connection",
    "notification": "For notification problems:\n1. Check app permissions\n2. Verify notification settings\n3. Restart the app",
    # Add all issue solutions...
}

# LangChain Tools
def content_creation_tool_func(content_type: str) -> dict:
    """Provides detailed guidance about creating content on BigShorts"""
    std_content_type = content_type.lower()
    
    # Standardize content type
    for mapping_key, mapping_value in CONTENT_TYPE_MAPPING.items():
        if mapping_key in std_content_type:
            std_content_type = mapping_value
            break
    
    guide = CONTENT_GUIDES.get(std_content_type)
    
    if guide is None:
        return {
            "type": "error",
            "content": f"Guide for {content_type} not found. Available types: SHOT, SNIP, SSUP, Collab, Mini"
        }
    
    return {
        "type": "content_guide",
        "content": guide
    }

def handle_issue_tool_func(issue_type: str) -> str:
    """Handles common platform issues and provides solutions"""
    solution = ISSUE_SOLUTIONS.get(issue_type.lower())
    
    if solution is None:
        return f"I don't have a solution for that issue. Try asking about: {', '.join(ALLOWED_ISSUE_TYPES)}"
    
    return solution

def platform_guide_tool_func(section: str) -> str:
    """Provides guidance about different sections of the platform"""
    platform_sections = {
        "shot": "SHOT is our platform's photo sharing feature. Would you like me to show you how to create a SHOT?",
        "snip": "SNIP is our platform's short video feature (similar to reels). Would you like me to show you how to create a SNIP?",
        "ssup": "SSUP is our platform's stories feature for temporary 24-hour content. Would you like me to show you how to create a SSUP?",
        "collab": "Our collaboration features let you create content with other users. Would you like me to show you how to use collaboration features?",
        "Mini": "Mini is our platform's longer video format. Would you like me to show you how to create a Mini?",
        # Add more sections...
    }
    
    return platform_sections.get(section.lower(), 
        "I don't have information about that section. Try asking about 'SHOT', 'SNIP', 'SSUP', 'Mini', or 'collab'.")

def generate_interactive_ideas_func() -> str:
    """Generates random interactive video ideas for Snips"""
    ideas = [
        "Create a Snip where viewers can click a button to watch a related tutorial video.",
        "Add an interactive button that redirects viewers to a behind-the-scenes video.",
        "Create a Snip where viewers can choose which type of content they want to watch next.",
        "Embed a button that lets viewers explore different topics you've covered.",
        "Add a button that takes viewers to a Q&A session you've recorded.",
    ]
    
    return f"Here's an interactive idea for your Snip: {random.choice(ideas)}"

def get_trending_content_func() -> dict:
    """Returns trending content suggestions"""
    return {
        "type": "suggestion_buttons",
        "content": {
            "message": "Check out what's trending on BigShorts! 📈",
            "buttons": [
                {"text": "Check Trending Snips", "action": "redirect", "destination": "/trending/snips"},
                {"text": "Discover Popular Creators", "action": "redirect", "destination": "/trending/creators"},
                {"text": "See Trending Shots", "action": "redirect", "destination": "/trending/shots"}
            ]
        }
    }

# Create LangChain Agent
class BigShortsAgent:
    def __init__(self, model_path: str):
        """Initialize the LangChain-based autonomous agent"""
        
        # Initialize LLM with callbacks
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            n_threads=8,
            n_batch=512,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
            callback_manager=callback_manager,
            verbose=False,
        )
        
        # Define tools
        self.tools = [
            StructuredTool(
                name="content_creation_guide",
                description="Use this tool when user asks about creating content like SHOT, SNIP, SSUP, Collab, or Mini. Provides step-by-step visual guides.",
                func=content_creation_tool_func,
                args_schema=ContentGuideInput
            ),
            StructuredTool(
                name="handle_issue",
                description="Use this tool when user has problems or issues with the platform (login, upload, notification, etc.)",
                func=handle_issue_tool_func,
                args_schema=IssueInput
            ),
            Tool(
                name="platform_guide",
                description="Use this tool to explain what different platform features are (SHOT, SNIP, SSUP, etc.)",
                func=platform_guide_tool_func
            ),
            Tool(
                name="interactive_ideas",
                description="Use this tool when user asks for ideas for interactive Snip videos",
                func=generate_interactive_ideas_func
            ),
            Tool(
                name="trending_content",
                description="Use this tool when user asks about trending or popular content on BigShorts",
                func=get_trending_content_func
            ),
        ]
        
        # Create custom prompt template
        self.prompt = PromptTemplate(
            template="""You are Gyan.Ai, the helpful assistant for BigShorts social media platform.
Your role is to help users create content (SHOT photos, SNIP videos, SSUP stories, Mini long-form videos, Collab posts) 
and solve platform issues.

IMPORTANT RULES:
1. Only help with BigShorts platform features
2. For off-topic questions, politely redirect to BigShorts features
3. Be concise and friendly
4. Use tools to provide detailed guides when users want to create content
5. Use tools to troubleshoot issues when users have problems

Available tools:
{tools}

Tool names: {tool_names}

Use this format:
Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final response to the user

Current conversation:
{chat_history}

Question: {input}
{agent_scratchpad}""",
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def process_query(self, user_input: str) -> Union[str, dict]:
        """Process user query through the agent"""
        
        # Handle greetings
        greetings = ["hello", "hi", "hey", "greetings", "howdy"]
        if user_input.lower().strip() in greetings:
            responses = [
                "Hello! 😀 Welcome to BigShorts! Ready to create some awesome content today?",
                "Hey there! 😃 I'm Gyan.Ai, your BigShorts assistant. What would you like to create?",
                "Hi! 😊 Looking to make a SHOT, SNIP, SSUP, or MINI on BigShorts today?",
            ]
            
            faqs = [
                {"question": "How to create a MINI?", "content_type": "Mini"},
                {"question": "How do I create a SHOT?", "content_type": "shot"},
                {"question": "How do I create a SNIP?", "content_type": "snip"},
                {"question": "How do I create a SSUP?", "content_type": "ssup"},
            ]
            
            return {
                "type": "greeting_with_faqs",
                "content": {
                    "greeting": random.choice(responses),
                    "faqs": faqs
                }
            }
        
        # Handle FAQ selections
        if user_input.startswith("FAQ:"):
            content_type = user_input.split("FAQ:")[1].strip()
            return content_creation_tool_func(content_type)
        
        # Run through agent
        try:
            result = self.agent_executor.invoke({"input": user_input})
            
            # Extract final answer
            output = result.get("output", "I couldn't process that request.")
            
            # Check if output is already structured
            if isinstance(output, dict):
                return output
            
            return {"type": "message", "content": output}
            
        except Exception as e:
            print(f"Agent error: {str(e)}")
            return {
                "type": "error",
                "content": "I encountered an issue. Can I help you with creating SHOT, SNIP, SSUP, Mini, or Collab content?"
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.memory.chat_memory.messages


# Interactive demo
def run_agent():
    """Run the agent in interactive mode"""
    model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download the model first.")
        return
    
    try:
        print("Initializing BigShorts Agent...")
        agent = BigShortsAgent(model_path)
        print("\nBigShorts Agent ready! Type 'exit' to quit.\n")
        
        print("Agent: Hi! I'm Gyan.Ai, your BigShorts assistant. How can I help you today?")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nAgent: Thanks for chatting! Have a great day on BigShorts!")
                break
            
            if not user_input:
                continue
            
            # Process query
            response = agent.process_query(user_input)
            
            # Display response
            if isinstance(response, dict):
                if response.get("type") == "message":
                    print(f"\nAgent: {response.get('content')}")
                elif response.get("type") == "content_guide":
                    guide = response.get("content", {})
                    print(f"\nAgent: Here's the guide for {guide.get('title', 'content creation')}!")
                    print("(Visual guide would be displayed in the app)")
                elif response.get("type") == "greeting_with_faqs":
                    print(f"\nAgent: {response['content']['greeting']}")
                    print("\nPopular questions:")
                    for faq in response['content']['faqs']:
                        print(f"  - {faq['question']}")
                else:
                    print(f"\nAgent: {response}")
            else:
                print(f"\nAgent: {response}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_agent()
