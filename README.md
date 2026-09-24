# 🧩 LangGraph Patterns Playground

A hands-on exploration of **LangGraph's core building blocks** — sequential pipelines, parallel state reducers, conditional routing, tool-calling loops, and human-in-the-loop workflows — built while learning agentic AI system design. Each folder isolates one architectural pattern, and two of them are shipped as full applied products with Streamlit UIs on top of their CLI logic.

---

## 📖 Overview

Agentic AI systems are built from a small set of recurring control-flow patterns. Rather than jumping straight into one large project, this repo was built to deliberately isolate and understand each pattern on its own — so it can be recognized and reused correctly when designing larger, production-style agent systems.

| # | Pattern | Folder | What It Demonstrates |
|---|---------|--------|----------------------|
| 1 | **Sequential Pipeline** | [`sequential-pipeline/`](./sequential-pipeline) | Linear multi-stage graph where each node transforms and passes state to the next (edit → script → translate). |
| 2 | **Parallel Execution + Reducers** | [`parallel-reducers/`](./parallel-reducers) | Fan-out from `START` to multiple independent nodes running concurrently, with a custom reducer function safely merging their outputs into shared state. |
| 3 | **Conditional Routing (RAG Router)** | [`college-assistant/`](./college-assistant) | An LLM-based classifier node that dynamically routes a query to one of several retrieval paths using `add_conditional_edges`. Shipped as both a CLI and a Streamlit app. |
| 4 | **Iterative Tool Use + Self-Review Loop** | [`iterative-tool-use/`](./iterative-tool-use) | A writer LLM bound to a web-search tool, paired with a separate reviewer LLM that approves/rejects drafts, looping back with feedback until approval or a max-attempt cap. Shipped as both a CLI and a Streamlit app. |
| 5 | **Human-in-the-Loop (HITL)** | [`human-in-the-loop/`](./human-in-the-loop) | Graph execution paused mid-run using `interrupt()` and resumed with `Command(resume=...)`, driven by a checkpointer so state persists across the pause. |
| 6 | **State Schema Patterns** | [`state-patterns/`](./state-patterns) | Four different ways to define LangGraph state — `TypedDict`, `Pydantic` (with runtime validation), `dataclass`, and the built-in `MessagesState`. |

---

## 🎓 Applied Project 1: College Assistant

Combines routing + dual RAG pipelines into a working product:

- **Query classification** — an LLM node tags each question as `academic`, `fee`, or `general`
- **Conditional routing** — the graph dynamically sends the query to the matching retriever
- **Dual RAG pipelines** — separate FAISS vector stores built from an academic handbook and a fee-structure document
- **Personalized responses** — answers are tailored to the student's selected programme (BCA / BBA / B.Com(H))
- **Streamlit interface** — chat UI with programme selection, chat history, and a badge showing which route answered the query
- **CLI version included** — a terminal-based variant of the same graph for comparison

```
User Query → Classifier Node → ┬─ Academic RAG ─┐
                                ├─ Fee RAG ───────┼─→ Response Node → Answer
                                └─ General ───────┘
```

## ✍️ Applied Project 2: LinkedIn Post Generator (Writer + Reviewer Agent)

Combines iterative tool-calling with a self-review loop:

- **Writer agent** — drafts a LinkedIn post, and can call a live web-search tool first if the topic needs current information
- **Reviewer agent** — a separate, low-temperature LLM grades the draft against 7 fixed criteria (hook, length, CTA, tone, etc.) and returns an approve/reject verdict with feedback
- **Feedback loop** — rejected drafts are sent back to the writer with the reviewer's notes, up to 3 attempts
- **Streamlit interface** — live status updates per step (drafting / searching / reviewing), an expandable history of every attempt with its verdict, and a downloadable final post
- **CLI version included** — a terminal-based variant of the same graph for comparison

```
START → Writer ─┬─ needs search → Tools ─┐
                 └─ no search ─→ Extract  ┴→ Reviewer ─┬─ approved / max attempts → END
                                                        └─ rejected → back to Writer
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph), LangChain |
| LLMs | Groq (`openai/gpt-oss-120b`), OpenRouter |
| Embeddings & Retrieval | HuggingFace Sentence Transformers, FAISS |
| Tools | Tavily Search API |
| UI | Streamlit |
| Environment | Python, `python-dotenv` |

---

## 📂 Repository Structure

```
langgraph-patterns-playground/
│
├── sequential-pipeline/
│   └── sequential_base.py
│
├── parallel-reducers/
│   └── parallel_reducers.py
│
├── college-assistant/
│   ├── app.py                  # Streamlit version
│   ├── conditional_rag.py      # CLI version
│   ├── academics_handbook.pdf
│   └── fee_structure.pdf
│
├── iterative-tool-use/
│   ├── app.py                  # Streamlit version
│   └── iterative_tools.py      # CLI version
│
├── human-in-the-loop/
│   └── human_in_the_loop.py
│
├── state-patterns/
│   └── states.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/alfezkhan-ds/langgraph-patterns-playground.git
cd langgraph-patterns-playground

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

## ▶️ Running Each Pattern

```bash
# Sequential pipeline
python sequential-pipeline/sequential_base.py

# Parallel reducers
python parallel-reducers/parallel_reducers.py

# College Assistant — CLI
python college-assistant/conditional_rag.py

# College Assistant — Streamlit
streamlit run college-assistant/app.py

# LinkedIn Post Generator — CLI
python iterative-tool-use/iterative_tools.py

# LinkedIn Post Generator — Streamlit
streamlit run iterative-tool-use/app.py

# Human-in-the-loop
python human-in-the-loop/human_in_the_loop.py

# State schema patterns (reference only — inspect the file)
python state-patterns/states.py
```

---

## 🧠 Key Learnings

- Designing **state schemas** and choosing the right reducer strategy for concurrent writes
- Difference between **static edges** and **conditional edges** for dynamic routing
- Using `interrupt()` and checkpointers to build **pausable, resumable** agent workflows
- Combining **retrieval, classification, and generation** into a coherent multi-node graph
- Structuring an agent's **self-review loop** with a distinct evaluator LLM and clear stop conditions
- Streaming graph execution (`stream_mode="updates"`) to drive **live, step-by-step UI feedback** instead of waiting on a single blocking `invoke()` call

---

## 👤 Author

**Alfez Khan**
Data Scientist | AI/ML & Gen AI Engineering

- LinkedIn: [linkedin.com/in/alfezkhan-ds](https://linkedin.com/in/alfezkhan-ds)
- Kaggle: [kaggle.com/alfezkhan71](https://kaggle.com/alfezkhan71)

---

*This repository is a personal learning project built while exploring agentic AI system design with LangGraph. Feedback and suggestions are welcome.*
