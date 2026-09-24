# Import required standard libraries and LangGraph/LangChain modules
import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
# Load environment variables from .env file
load_dotenv()


# tools(without using langchain decorator)
# Initialize Tavily search tool and limit to top 3 results
search_tool = TavilySearch(max_results=3)

tools = [search_tool]

# llm(needed to use 2 llm,,1 in generator,1 in reviewr process,,)

# (writer)
# Setup writer LLM using OpenRouter and bind the search tool to it
writer_llm = ChatOpenRouter(
    model="openrouter/free", temperature=0.7)
write_llm_with_tools = writer_llm.bind_tools(
    tools)  # binding llm with search tool

# reviewer
# Setup strict reviewer LLM using Groq with low temperature for objective evaluation
reviewer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

# State Building

# Define the global state schema to hold iteration data across graph nodes


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# nodes

# System prompt instructing the writer on format, constraints, and iteration
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state: State) -> dict:
    """ Write or rewrites the LinkedIn post. Can call Tavily to search first"""
    # Track number of writing attempts
    attempt = state.get("attempt", 0) + 1
    topic = state['topic']
    previous_feedback = state['review_feedback']

    # Dynamically build the prompt based on whether it is a first draft or revision
    if attempt == 1:
        user_message = (
            f"write a linkedin post on this {topic}"
            f"if you need current info search the web first"
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"here is the reviewer's feedback \n\n\n{previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentioned"
            f"do not repeat the same mistake"
        )
    # Invoke writer LLM with current context
    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
    response = write_llm_with_tools.invoke(messages)

    # Return updated messages array and attempt count
    return {
        "messages": [("human", user_message), response],
        "attempt": attempt
    }


# Prebuilt LangGraph node to execute any tool calls requested by the LLM
tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    """ After the writer finishes tool calls, pull the final text out as draft"""
    # Grab the text generated in the last LLM response
    last_message = state['messages'][-1]
    draft = last_message.content
    print(f"\n\n generated post \n {draft}\n ")
    return {"draft": draft}


# System prompt dictating strict evaluation criteria for the reviewer
REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

# make reviewer(feedback) node


def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['draft']

    # Prompt the reviewer LLM with the latest draft
    promt = (f"review this LinkedIn post draft : \n"
             f"{draft}\n"
             f"give your reviews"
             )
    response = reviewer_llm.invoke([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", promt)
    ])
    review_text = response.content.strip()

    # Parse LLM output to determine boolean approval status
    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    # Extract detailed feedback text
    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    # Update state with verdict and reviewer notes
    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }

# we need router function to decide after is_approved false,,for loopback from reiwer to writer again with feedback using.
# so for that decision we use router like we used in conditional

# Router Function
# using it for tool call loop b/w writer llm and search tool


def should_use_tool(state: State):
    # Route to tools node if writer requested a tool call, else proceed to extract draft
    last_message = state['messages'][-1]

    if getattr(last_message, 'tool_calls', None):
        return "tools"
    return "extract_draft"


# using it for feedback loop bw reviewr nd writer
def should_stop_looping(state: State):
    # Terminate workflow if approved or if maximum loop limit (3) is reached
    if state['is_approved']:
        print("post has been approved\n")
        return END
    if state["attempt"] >= 3:
        print("reached max attempts")
        return END
    # Otherwise route back to writer for revisions
    return "writer"


# build the graph
# Initialize LangGraph with our custom State schema
graph = StateGraph(State)

# Add all components as executable nodes
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)

# Define entry point
graph.add_edge(START, "writer")

# Conditional routing from writer to tools or extraction
graph.add_conditional_edges(
    "writer", should_use_tool,
)

# Connect tools and extraction nodes directly to the reviewer
graph.add_edge("tools", "reviewer")
graph.add_edge("extract_draft", "reviewer")

# Conditional routing from reviewer back to writer or terminating the process
graph.add_conditional_edges(
    "reviewer", should_stop_looping
)

# Compile graph into runnable application
app = graph.compile()

# CLI User Interface Setup
print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")

print("=" * 55)

# Prompt user for topic input
topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    # Define initial baseline state
    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    # Execute workflow graph
    final_state = app.invoke(initial_state)

    # Print ultimate results and metadata
    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}")
