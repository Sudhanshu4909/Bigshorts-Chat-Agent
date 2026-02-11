# BigShorts LangChain Autonomous Agent

Modern AI assistant for BigShorts social media platform, built with LangChain for autonomous task handling.

## 🚀 Features

### Basic Agent (`bigshorts_langchain_agent.py`)
- ✅ LangChain-based autonomous reasoning
- ✅ Tool-based architecture for content creation guides
- ✅ Issue resolution system
- ✅ Memory-enabled conversations
- ✅ Structured tool inputs with Pydantic
- ✅ ReAct agent pattern

### Advanced Agent (`advanced_bigshorts_agent.py`)
- ✅ All basic features PLUS:
- ✅ **RAG (Retrieval-Augmented Generation)** with vector store
- ✅ **Multi-session management** (concurrent users)
- ✅ **Analytics & metrics** tracking
- ✅ **Session persistence**
- ✅ **Enhanced context** (4K tokens)
- ✅ **GPU acceleration** support

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements_langchain.txt
```

### 2. Download LLM Model

```bash
# Create models directory
mkdir -p models

# Download Mistral 7B (recommended)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  -O models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 3. (Optional) For RAG Features

```bash
pip install faiss-cpu sentence-transformers
# OR
pip install chromadb
```

## 🎯 Usage

### Basic Agent

```python
from bigshorts_langchain_agent import BigShortsAgent

# Initialize
agent = BigShortsAgent("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Process queries
response = agent.process_query("How do I create a SHOT?")
print(response)

# Get conversation history
history = agent.get_conversation_history()
```

### Run Interactive Demo

```bash
python bigshorts_langchain_agent.py
```

### Advanced Agent with Sessions

```python
from advanced_bigshorts_agent import AdvancedBigShortsAgent

# Initialize with RAG
agent = AdvancedBigShortsAgent(
    "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    enable_rag=True
)

# Multi-user support
response1 = agent.process_query("How to create SNIP?", session_id="user_123")
response2 = agent.process_query("Login issues", session_id="user_456")

# Get analytics
analytics = agent.get_analytics()
print(analytics)

# Save analytics
agent.save_analytics("analytics.json")
```

### Run Advanced Demo

```bash
python advanced_bigshorts_agent.py
```

## 🛠️ Available Tools

The agent has access to these tools:

1. **content_creation_guide** - Step-by-step guides for:
   - SHOT (photos)
   - SNIP (short videos)
   - SSUP (stories)
   - Mini (long-form videos)
   - Collab (collaborative posts)

2. **handle_issue** - Solves platform issues:
   - Login problems
   - Upload errors
   - Notification issues
   - Privacy settings
   - And more...

3. **platform_guide** - Explains features and sections

4. **interactive_ideas** - Creative content ideas

5. **trending_content** - Popular content recommendations

6. **knowledge_search** (Advanced only) - RAG-powered knowledge base search

## 📊 How It Works

### ReAct Agent Pattern

The agent uses the ReAct (Reasoning + Acting) pattern:

```
Question: How do I create a SHOT?
  ↓
Thought: User wants to create photo content
  ↓
Action: content_creation_guide
  ↓
Action Input: {"content_type": "shot"}
  ↓
Observation: Retrieved step-by-step guide
  ↓
Thought: I have the guide information
  ↓
Final Answer: [Returns structured guide with images]
```

### Architecture

```
User Input
    ↓
┌─────────────────┐
│  LangChain      │
│  Agent          │
│  (ReAct)        │
└────────┬────────┘
         ↓
    ┌────────┐
    │ Tools  │ ← content_creation_guide
    │        │ ← handle_issue
    │        │ ← platform_guide
    │        │ ← interactive_ideas
    │        │ ← trending_content
    │        │ ← knowledge_search (RAG)
    └────────┘
         ↓
    Response
```

## 🔧 Configuration

### Adjust LLM Parameters

```python
self.llm = LlamaCpp(
    model_path=model_path,
    n_ctx=4096,          # Context window
    n_gpu_layers=-1,     # GPU layers (-1 = all)
    temperature=0.7,     # Creativity (0.0-1.0)
    max_tokens=512,      # Max response length
    top_p=0.95,          # Nucleus sampling
)
```

### Enable/Disable RAG

```python
agent = AdvancedBigShortsAgent(
    model_path,
    enable_rag=False  # Disable RAG
)
```

## 📈 Analytics

The advanced agent tracks:
- Total queries processed
- Tool usage frequency
- Popular content types
- Common issues
- Session statistics

Access analytics:

```python
analytics = agent.get_analytics()
# {
#   "total_queries": 150,
#   "tool_usage": {"content_creation_guide": 45, ...},
#   "popular_content_types": {"shot": 20, "snip": 25},
#   "active_sessions": 3
# }
```

## 🎨 Response Types

The agent returns structured responses:

```python
# Message response
{
    "type": "message",
    "content": "Here's how to create a SHOT..."
}

# Content guide with steps
{
    "type": "content_guide",
    "content": {
        "title": "Creating a BigShorts SHOT",
        "steps": [...]
    }
}

# Greeting with FAQs
{
    "type": "greeting_with_faqs",
    "content": {
        "greeting": "Hello! ...",
        "faqs": [...]
    }
}

# Error response
{
    "type": "error",
    "content": "I encountered an issue..."
}
```

## 🔍 Example Interactions

```python
# Content creation
agent.process_query("How to create a SNIP?")
# → Returns step-by-step guide with images

# Issue resolution
agent.process_query("I can't upload videos")
# → Returns troubleshooting steps

# General inquiry
agent.process_query("What is SSUP?")
# → Explains feature and offers guide

# Creative ideas
agent.process_query("Give me interactive video ideas")
# → Returns creative suggestions
```

## 🚧 Extending the Agent

### Add New Tool

```python
def my_custom_tool(param: str) -> str:
    """Custom tool implementation"""
    return f"Processed: {param}"

# Add to tools list
new_tool = Tool(
    name="custom_tool",
    description="Does something custom",
    func=my_custom_tool
)

agent.tools.append(new_tool)
```

### Add to Knowledge Base (RAG)

```python
from langchain.docstore.document import Document

new_doc = Document(
    page_content="New feature information...",
    metadata={"type": "feature", "name": "new_feature"}
)

agent.vectorstore.add_documents([new_doc])
```

## 🎯 Key Improvements Over Original

| Feature | Original | LangChain Agent |
|---------|----------|-----------------|
| Architecture | Monolithic functions | Tool-based modular |
| Reasoning | Hardcoded logic | Autonomous ReAct |
| Memory | Simple list | Structured ConversationMemory |
| Scalability | Single session | Multi-session support |
| Knowledge | Hardcoded | RAG with vector search |
| Analytics | None | Comprehensive tracking |
| Extensibility | Difficult | Easy tool addition |

## ⚡ Performance Tips

1. **Use GPU**: Set `n_gpu_layers=-1` for 10x faster inference
2. **Adjust context**: Lower `n_ctx` if memory constrained
3. **Batch queries**: Process multiple queries in same session
4. **Cache embeddings**: RAG embeddings computed once at startup
5. **Limit iterations**: Set `max_iterations=3` for faster responses

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Check model file exists
ls -lh models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Try smaller model if memory issues
# Download Q3 or Q2 quantization instead
```

### RAG Not Working
```bash
# Install dependencies
pip install faiss-cpu sentence-transformers

# Or disable RAG
agent = AdvancedBigShortsAgent(model_path, enable_rag=False)
```

### Slow Responses
- Reduce `n_ctx` to 2048
- Use quantized model (Q4 or Q3)
- Set `max_iterations=3`
- Disable verbose mode




