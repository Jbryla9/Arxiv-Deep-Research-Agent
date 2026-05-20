# ArXiv Deep Research Agent

An agentic AI system that conducts multi-step academic research by combining
web search (Tavily), arXiv paper retrieval, vector similarity search (Weaviate),
and a hierarchical multi-agent architecture built with LangGraph.

## Example Output

**Query:** "Supervised Learning"

**Agent Response:**
> Supervised learning is a type of machine learning where an algorithm learns
> to map input data to specific outputs based on labeled example pairs...
>
> **References found by agent:**
> - *Learning to Impute: A General Framework for Semi-supervised Learning* — [PDF](https://arxiv.org/pdf/1912.10364v3)
> - *SelfMatch: Combining Contrastive Self-Supervision and Consistency for Semi-Supervised Learning* — [PDF](https://arxiv.org/pdf/2101.06480v1)

---

## Architecture

The system uses a two-level agent hierarchy:

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Main Agent              │
│  (GPT-4o-mini, LangGraph)       │
│  - Manages TODOs                │
│  - Manages virtual file system  │
│  - Delegates to sub-agents      │
└────────────┬────────────────────┘
             │ delegates via task()
             ▼
┌─────────────────────────────────┐
│       Research Sub-Agent        │
│  - Searches web (Tavily)        │
│  - Searches arXiv papers        │
│  - Reflects via think_tool      │
└─────────────────────────────────┘
```

### Main Agent Tools
| Tool | Purpose |
|------|---------|
| `task()` | Delegate research to sub-agent |
| `think_tool()` | Strategic reflection and planning |
| `ls()` | List virtual filesystem |
| `read_file()` | Read saved research files |
| `write_file()` | Save notes to virtual filesystem |
| `write_todos()` | Create research plan |
| `read_todos()` | Check progress against plan |

### Research Sub-Agent Tools
| Tool | Purpose |
|------|---------|
| `limited_arxiv_search()` | Search arXiv (max 5 calls/task) |
| `tavily_search()` | Web search with content summarization |
| `think_tool()` | Reflect on findings, plan next steps |

---

## How It Works — Step by Step

### Step 1: Query Intake & Planning
The main agent receives the user's query and immediately calls `write_todos()`
to create a structured research plan. For example, for "Random Forest machine
learning" it plans: (1) delegate web research, (2) retrieve arXiv papers,
(3) synthesize and respond.

### Step 2: Task Delegation
The main agent calls `task(description, subagent_type="research-agent")` to
spin up an isolated research sub-agent. For complex queries, up to 2 sub-agents
run in parallel. Each sub-agent receives a complete, standalone instruction —
sub-agents cannot see each other's work.

### Step 3: Web Search (Tavily)
The sub-agent calls `tavily_search(query)` which:
1. Queries the Tavily API for results
2. Fetches full page content via `httpx`
3. Converts HTML to Markdown via `markdownify`
4. Summarizes content using `gpt-4o-mini` with structured output
5. Saves full content + summary to the virtual filesystem
6. Returns a brief overview to the agent (not the full content)

This keeps the agent's context window small — it reads files only when needed.

### Step 4: arXiv Paper Retrieval
The sub-agent calls `limited_arxiv_search(query)` which:
1. Queries the arXiv API (`export.arxiv.org/api/query`) with a 3-second
   delay to respect rate limits, with automatic retry on 429 errors
2. Parses the Atom XML feed to extract title, abstract, authors, DOI,
   categories, published date, and PDF URL
3. Encodes each paper's title + abstract using
   `sentence-transformers/all-MiniLM-L6-v2`
4. Stores embeddings in a local **Weaviate** vector database
5. Retrieves the most relevant paper via cosine similarity search
6. Saves the abstract and metadata to the virtual filesystem

### Step 5: Reflection with think_tool
After each search the sub-agent calls `think_tool(reflection)` to explicitly
reason about: what was found, what is still missing, whether enough sources
exist to answer the question, and whether to keep searching or stop.
This enforces deliberate, quality-controlled research.

### Step 6: Synthesis
Once the sub-agent finishes, the main agent:
1. Reads the TODO list to track progress
2. Uses `read_file()` to load saved research files
3. Synthesizes a structured answer with sections for definition, mechanisms,
   applications, and challenges
4. Appends a **References** section with arXiv paper titles and PDF links
5. Adds a fun historical fact about the year the paper was published

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph + LangChain |
| LLM | OpenAI GPT-4o-mini |
| Web search | Tavily API |
| Academic search | arXiv API (Atom feed, custom implementation) |
| Vector database | Weaviate (local Docker) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| HTML→Markdown | `markdownify` |
| Observability | LangSmith |

---

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop
- API keys for: OpenAI, Tavily, LangSmith (optional)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/arxiv-deep-research-agent.git
cd arxiv-deep-research-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Start Weaviate
```bash
docker-compose up -d
```

Verify it's running:
```bash
docker ps --format "table {{.Names}}\t{{.Ports}}"
# Should show: arxiv_project-weaviate-1   0.0.0.0:8080->8080/tcp
```

### 5. Run the notebook
```bash
jupyter notebook notebooks/arxiv_deep_research_agent.ipynb
```

---

## Project Structure

```
arxiv_project/
├── notebooks/
│   └── arxiv_deep_research_agent.ipynb   # Main notebook
├── src/
│   ├── research_tools_arxiv.py           # Tavily + arXiv tools
│   ├── file_tools_arxiv.py               # Virtual filesystem tools
│   ├── todo_tools_arxiv.py               # TODO management tools
│   ├── task_tool_arxiv.py                # Sub-agent task delegation
│   ├── state_arxiv.py                    # LangGraph state definition
│   ├── prompts_arxiv.py                  # All agent prompts
│   └── utils.py                          # Helper utilities
├── docker-compose.yml                    # Weaviate configuration
├── .env.example                          # Environment variable template
├── .gitignore
└── README.md
```

---

## Key Design Decisions

**Why a virtual filesystem?** Keeping raw search content out of the agent's
context window prevents token overflow. The agent stores files and reads them
selectively — only when needed to answer the question.

**Why Weaviate for arXiv?** Semantic similarity search finds the most relevant
paper even when keyword matching fails. The embedding model runs locally
(no API cost).

**Why rate-limit arXiv calls?** The arXiv API enforces rate limits and returns
HTTP 429 if called too frequently. The 3-second delay and exponential backoff
retry logic (10s, 20s) make the tool robust in multi-agent parallel execution.

**Why gpt-4o-mini for summarization?** Summarization runs on every web page
fetched. Using a cheaper, faster model here keeps costs low while the main
reasoning uses the same model for consistency.

---

## Limitations

- arXiv search is limited to 5 calls per research task to prevent abuse
- Tavily search is limited to 5 calls per research task  
- Weaviate collection is recreated on each search (stateless by design)
- The fun fact year is currently randomized, not parsed from actual paper dates