import streamlit as st
from typing import TypedDict, Annotated
import os
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
# this vector storage works on ram not saves vector in loacl
from langchain_community.vectorstores import FAISS
load_dotenv()

# ============================================================
# ORIGINAL LOGIC — UNCHANGED
# ============================================================

# Step 1 - RAG Retriever building
# Downloads the embedding model from HuggingFace to convert words into
# mathematical coordinate arrays (vectors) representing their true meaning.


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")


embeddings = get_embeddings()

# function for retriever


def build_retriver(pdf_path: str):
    # PyPDFLoader extracts all lines of text page-by-page from the raw PDF file.
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    # Breaks giant pages into smaller, manageable chunks of 800 characters.
    # The 100-character overlap prevents sentences from being awkwardly cut in half.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100)

    chunks = splitter.split_documents(document)  # made chunks of doc

    # The magic line: Sends text chunks to our embedding model to generate vectors,
    # then instantly builds an in-RAM FAISS vector index database matching text to numbers.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Converts the raw database into a searchable tool that will pull up the top 4
    # most textually relevant chunks matching a student's question.
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner="Indexing college documents (this runs once)...")
def load_retrievers():
    # Builds two independent, distinct search engines containing separate knowledge domains.
    academic = build_retriver("academics_handbook.pdf")
    fee = build_retriver("fee_structure.pdf")
    return academic, fee


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


# step 2 - lets create State
# This defines the shared, central data graph object (the memory blueprint).
class State(TypedDict):
    # Normal string field: Records and overwrites student selection
    programme: str
    # Special list: Uses LangGraph reducer to auto-append history
    messages: Annotated[list, add_messages]
    # Normal string field: Overwrites with router classification tags
    query_type: str
    # Normal string field: Overwrites with text snippets from PDFs
    retrived_context: str

# Step 3 - create nodes


def make_classifier_node(llm):
    def classifier_node(state: State) -> dict:
        """Look at the latest user message and decide which path to take."""

        # Fetches the text string of the absolute newest question typed by the user.
        last_message = state['messages'][-1].content

        # Prepares a specialized instruction telling the LLM to function strictly
        # as a single-word classifier, giving it direct rules for each possible path.
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )

        response = llm.invoke(prompt)
        category = response.content.strip().lower()

        # Safety checks to sanitize and normalize the text returned by the LLM
        # so it perfectly aligns with our routing expected outputs.
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"query_type": category}
    return classifier_node


def make_academic_rag_node(academic_retriver):
    def academic_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the academics handbook."""
        query = state["messages"][-1].content
        # Searches the academic vector database using semantic similarity matching.
        docs = academic_retriver.invoke(query)
        # Extracts text components from document chunks and stitches them into a single string.
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrived_context": context}
    return academic_rag_node


def make_fee_rag_node(fee_retriver):
    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the fee structure PDF."""
        query = state["messages"][-1].content
        # Searches the fee structure vector database using semantic similarity matching.
        docs = fee_retriver.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrived_context": context}
    return fee_rag_node


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    # Sets a clear text marker flag so the subsequent response node skips document parsing.
    return {"retrived_context": "NO_RETRIEVAL_NEEDED"}


def make_response_node(llm):
    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrived_context"]

        # branch logic checks if context contains a marker flag or active document text snippets.
        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            # Dynamically targets the instructions, feeding the extracted PDF context
            # and student programme directly to the prompt layout for factual answers.
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)
        # Appends the new AI generated message into the existing dialogue array in State via add_messages.
        return {"messages": [("ai", response.content.strip())]}
    return response_node


# step 4 - router function
def route_query(state: State):
    # This conditional logic matches string outputs directly to the nodes registered below.
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


# step 5 - Building the graph
@st.cache_resource(show_spinner="Building the assistant graph...")
def build_app():
    academic_retriver, fee_retriver = load_retrievers()
    llm = get_llm()

    graph = StateGraph(State)

    # Registers our functional programming blocks inside the LangGraph engine.
    graph.add_node("classifier", make_classifier_node(llm))
    graph.add_node("academic_rag", make_academic_rag_node(academic_retriver))
    graph.add_node("fee_rag", make_fee_rag_node(fee_retriver))
    graph.add_node("general", general_node)
    graph.add_node("response", make_response_node(llm))

    # edges

    # Defines the static starting execution sequence.
    graph.add_edge(START, "classifier")

    # Sets up a dynamic split decision. The system runs the function 'route_query'
    # right after 'classifier' node outputs data, sending processing downstream.
    graph.add_conditional_edges(
        "classifier", route_query
    )

    # All separate informational branches converge back to this final node
    # to stitch context together into human text responses.
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

# ---- minimal styling ----
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        background-color: #eef2ff;
        color: #4338ca;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROGRAMME_MAP = {
    "BCA": "BCA",
    "BBA": "BBA",
    "B.Com (H)": "B.Com (H)",
}

QUERY_TYPE_LABEL = {
    "academic": "📘 Academic",
    "fee": "💰 Fee",
    "general": "💬 General",
}

# ---- session state ----
if "programme" not in st.session_state:
    st.session_state.programme = None
if "messages" not in st.session_state:
    # list of {"role": ..., "content": ..., "tag": ...}
    st.session_state.messages = []
if "graph_messages" not in st.session_state:
    # raw (role, content) tuples fed to the graph
    st.session_state.graph_messages = []

# ---- sidebar ----
with st.sidebar:
    st.header("🎓 College Assistant")
    st.caption("Ask about academics, fees, or anything else.")

    st.divider()

    choice = st.radio(
        "Your programme",
        list(PROGRAMME_MAP.keys()),
        index=list(PROGRAMME_MAP.keys()).index(
            st.session_state.programme) if st.session_state.programme in PROGRAMME_MAP else 0,
    )
    if st.button("Confirm programme", use_container_width=True):
        st.session_state.programme = PROGRAMME_MAP[choice]
        st.toast(f"Set as {st.session_state.programme} student", icon="✅")

    st.divider()

    if st.session_state.programme:
        st.success(f"Chatting as **{st.session_state.programme}** student")
    else:
        st.warning("Select and confirm your programme to start")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.graph_messages = []
        st.rerun()

    with st.expander("ℹ️ About"):
        st.write(
            "This assistant routes your question to the right knowledge "
            "source — the academics handbook, the fee structure document, "
            "or general knowledge — using a LangGraph pipeline with FAISS "
            "retrieval and Groq's `openai/gpt-oss-120b` model."
        )

# ---- main area ----
st.title("🎓 College Assistant")

if not st.session_state.programme:
    st.info(
        "👈 Pick your programme in the sidebar and hit **Confirm programme** to begin.")
    st.stop()

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tag"):
            st.markdown(
                f'<span class="badge">{QUERY_TYPE_LABEL.get(msg["tag"], msg["tag"])}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# chat input
user_query = st.chat_input(
    "Ask about attendance, exams, fees, scholarships...")

if user_query:
    # show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.graph_messages.append(("human", user_query))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                app = build_app()
                # Triggers entire workflow graph run manually, feeding the chosen program
                # configuration and the full chat history to kickstart the system.
                result = app.invoke({
                    "programme": st.session_state.programme,
                    "messages": st.session_state.graph_messages,
                })
                answer = result["messages"][-1].content
                query_type = result.get("query_type", "general")
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
                query_type = "general"

        st.markdown(
            f'<span class="badge">{QUERY_TYPE_LABEL.get(query_type, query_type)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tag": query_type})
    st.session_state.graph_messages.append(("ai", answer))
