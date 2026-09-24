# Import standard libraries and LangGraph/LangChain modules for Human-in-the-Loop workflow
import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# writer llm
# Initialize OpenRouter LLM for content generation
writer_llm = ChatOpenRouter(
    model="openrouter/free", temperature=0.7)

# state building


# Define the global state schema to track graph data across iterations
class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# System prompt with strict rules and constraints for the writer LLM
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Write engaging, professional "
    "LinkedIn posts about the given topic. "
    "Rules: strong hook in the first line, one clear takeaway, easy to skim "
    "with short paragraphs, roughly 150-200 words, end with an engaging "
    "question or CTA, no hashtags. "
    "If you receive feedback on a previous draft, address every point carefully."
)


# Node to generate the initial draft or rewrite based on human feedback
def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post."""
    # Increment and track attempt counter
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    print(f"\n[Attempt {attempt}] Writer is drafting the post...")

    # Build prompt based on whether it is a first draft or a revision
    if attempt == 1:
        user_message = f"Write a LinkedIn post on this topic: {topic}"
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n\n"
            f"Reviewer feedback:\n{previous_feedback}\n\n"
            f"Write a NEW improved LinkedIn post that fixes every issue mentioned."
        )

    # Invoke the LLM with system instructions and user message
    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
    response = writer_llm.invoke(messages)

    draft = response.content
    print(f"\n[Draft ready]\n{'-' * 55}\n{draft}\n{'-' * 55}")

    # Return updated draft and attempt count to the state
    return {
        "draft": draft,
        "attempt": attempt
    }


# Node to pause execution and wait for human input via LangGraph interrupt
def human_review_node(state: State) -> dict:
    """Pauses the graph and waits for the human to approve or give feedback."""
    print(f"\n[Reached human review — Attempt {state['attempt']}]")

    # Suspend graph and expose data/instructions to the user
    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
        "instruction": "Type 'approved' to accept, or type your feedback to request a rewrite."
    })

    response = human_response.strip()

    # Parse user response to determine approval status
    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {
            "is_approved": True,
            "review_feedback": "Approved by human."
        }
    else:
        return {
            "is_approved": False,
            "review_feedback": response
        }


# Router function to end workflow if approved/max attempts reached, or loop back
def should_stop_looping(state: State):
    if state['is_approved']:
        print("\n[Post approved by human. Ending workflow.]")
        return END
    if state['attempt'] >= 3:
        print("\n[Reached max 3 attempts. Ending with last draft.]")
        return END
    print(
        f"\n[Rejected. Looping back to writer for attempt {state['attempt'] + 1}...]")
    return "writer"


# build the graph
# Initialize the state graph
graph = StateGraph(State)

# Register nodes in the graph
graph.add_node("writer", writer_node)
graph.add_node("human_review", human_review_node)

# Define standard execution flow
graph.add_edge(START, "writer")
graph.add_edge("writer", "human_review")

# Define dynamic routing based on human review outcome
graph.add_conditional_edges(
    "human_review",
    should_stop_looping,
    {
        "writer": "writer",
        END: END,
    },
)

# Initialize memory saver required for pausing/resuming state
checkpointer = MemorySaver()
# Compile graph into runnable application
app = graph.compile(checkpointer=checkpointer)


# Command Line Interface setup
print("=" * 55)
print("Welcome to the LinkedIn Post Generator (HITL Edition)")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, show it to")
print("YOU for review, and rewrite based on your feedback.")
print("=" * 55)

# Get topic from user
topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    # Configure a thread ID to track this specific session's state history
    config = {"configurable": {"thread_id": "linkedin_session_1"}}

    # Define initial state values
    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    # Start the graph execution
    result = app.invoke(initial_state, config=config)

    # Loop to handle execution pauses, surfacing the draft to the user and capturing input
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
        print("=" * 55)
        print(interrupt_data["draft"])
        print("=" * 55)
        print(f"\n{interrupt_data['instruction']}")

        # Get human feedback from terminal
        human_input = input("\nYour response: ").strip()

        # Resume graph execution with the human's feedback injected via Command
        result = app.invoke(Command(resume=human_input), config=config)

    # Print final results and metadata
    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(result["draft"])
    print("=" * 55)
    print(f"Total attempts: {result['attempt']}")
    print(f"Approved by human: {result['is_approved']}")
