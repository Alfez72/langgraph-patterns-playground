from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
import os
from typing import TypedDict, Annotated  # using annotated for reducers
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

# lets create state
# This custom rule merges new dictionary keys into the old dictionary
# instead of deleting the old data.


def merge_score_dicts(existing: dict, newupdate: dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}  # Combines both dictionaries

# This is the shared memory object for the graph


class AnalyzerState(TypedDict):
    raw_text: str  # Normal field: New values will overwrite old values

    # Custom field: Uses 'merge_score_dicts' to safely combine scores
    # from different nodes without losing previous data.
    safety_scores: Annotated[dict[str, int], merge_score_dicts]

    # Annotated[DataType, Reducer] -> Python feature to declare type + attach extra behavior
    # dict[str, int] -> The Data Type: A dictionary with text keys and integer scores
    # merge_score_dicts -> The Reducer: Tells LangGraph to combine data instead of overwriting it

# lets create nodes


def toxicity_node(state: AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under our single state key
    return {"safety_scores": {"toxicity_level": score}}


def copyright_node(state: AnalyzerState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}


builder = StateGraph(AnalyzerState)

# add nodes to make graph
builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_check", copyright_node)
builder.add_node("culture_node", culture_node)

# edges
builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_check")
builder.add_edge(START, "culture_node")

# ending edges for final node(end)
builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_check", END)
builder.add_edge("culture_node", END)

app = builder.compile()

sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """


initial_state = {
    "raw_text": sample_script,
    "safety_scores": {}  # Initialized as an empty dictionary
}

final_state = app.invoke(initial_state)


print(final_state["safety_scores"])
