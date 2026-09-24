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


# Step 1 - RAG Retriever building
# Downloads the embedding model from HuggingFace to convert words into
# mathematical coordinate arrays (vectors) representing their true meaning.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

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


# Builds two independent, distinct search engines containing separate knowledge domains.
academic_retriver = build_retriver("academics_handbook.pdf")
fee_retriver = build_retriver("fee_structure.pdf")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

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


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    # Searches the academic vector database using semantic similarity matching.
    docs = academic_retriver.invoke(query)
    # Extracts text components from document chunks and stitches them into a single string.
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrived_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    # Searches the fee structure vector database using semantic similarity matching.
    docs = fee_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrived_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    # Sets a clear text marker flag so the subsequent response node skips document parsing.
    return {"retrived_context": "NO_RETRIEVAL_NEEDED"}


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

graph = StateGraph(State)

# Registers our functional programming blocks inside the LangGraph engine.
graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

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

app = graph.compile()

# step 6 - Run the code

print("welcome to the College assistant \n\n")

print("which programe are you in ")
print("1. BCA")
print("2. BBA")
print("3. B.com (H)")

choice = input("\nEnter 1, 2 or 3 ")

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}
student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You're set as a {student_programme} student.")

# Infinite conversational loop processing terminal input values continuously.
while True:
    user_query = input("You:  ")

    if user_query.lower() in ["exit", "quit"]:
        break

    # Triggers entire workflow graph run manually, feeding the chosen program configuration
    # and the brand-new chat message payload block to kickstart the system.
    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })

    print(f"Assistant : {result['messages'][-1].content}")
