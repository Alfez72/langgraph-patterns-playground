import os
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_tavily import TavilySearch

load_dotenv()

# ============================================================
# ORIGINAL LOGIC — UNCHANGED (from iterative_tools.py)
# ============================================================


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


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


@st.cache_resource(show_spinner=False)
def get_llms():
    writer_llm = ChatOpenRouter(model="openrouter/free", temperature=0.7)
    search_tool = TavilySearch(max_results=3)
    write_llm_with_tools = writer_llm.bind_tools([search_tool])
    reviewer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
    return write_llm_with_tools, search_tool, reviewer_llm


def make_writer_node(write_llm_with_tools):
    def writer_node(state: State) -> dict:
        attempt = state.get("attempt", 0) + 1
        topic = state["topic"]
        previous_feedback = state["review_feedback"]

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

        messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
        response = write_llm_with_tools.invoke(messages)

        return {
            "messages": [("human", user_message), response],
            "attempt": attempt,
        }
    return writer_node


def extract_draft_node(state: State) -> dict:
    last_message = state["messages"][-1]
    return {"draft": last_message.content}


def make_reviewer_node(reviewer_llm):
    def reviewer_node(state: State) -> dict:
        draft = state["draft"]
        promt = (
            f"review this LinkedIn post draft : \n"
            f"{draft}\n"
            f"give your reviews"
        )
        response = reviewer_llm.invoke([
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", promt),
        ])
        review_text = response.content.strip()

        is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

        if "FEEDBACK:" in review_text:
            feedback = review_text.split("FEEDBACK", 1)[1].strip()
        else:
            feedback = review_text

        return {"review_feedback": feedback, "is_approved": is_approved}
    return reviewer_node


def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        return END
    if state["attempt"] >= 3:
        return END
    return "writer"


@st.cache_resource(show_spinner="Building the writer/reviewer graph...")
def build_app():
    write_llm_with_tools, search_tool, reviewer_llm = get_llms()
    tool_node = ToolNode([search_tool])

    graph = StateGraph(State)
    graph.add_node("writer", make_writer_node(write_llm_with_tools))
    graph.add_node("tools", tool_node)
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", make_reviewer_node(reviewer_llm))

    graph.add_edge(START, "writer")
    graph.add_conditional_edges("writer", should_use_tool)
    graph.add_edge("tools", "reviewer")
    graph.add_edge("extract_draft", "reviewer")
    graph.add_conditional_edges("reviewer", should_stop_looping)

    return graph.compile()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="LinkedIn Post Generator — Writer/Reviewer Agent",
    page_icon="📝",
    layout="centered",
)

st.markdown(
    """
    <style>
    .verdict-approved {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-rejected {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .attempt-label {
        font-weight: 600;
        color: #4338ca;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 LinkedIn Post Generator")
st.caption("A writer agent (with web search) and a strict reviewer agent iterate until the post is publish-ready.")

with st.sidebar:
    st.header("⚙️ How it works")
    st.markdown(
        "1. **Writer** drafts a post — searches the web first if the topic needs fresh info\n"
        "2. **Reviewer** grades it against 7 criteria (hook, length, CTA, tone, etc.)\n"
        "3. If rejected, the writer revises using the reviewer's feedback\n"
        "4. Loops up to **3 attempts**, or until approved"
    )
    st.divider()
    st.header("🔑 Required environment variables")
    for var in ["GROQ_API_KEY", "TAVILY_API_KEY", "OPENROUTER_API_KEY"]:
        present = bool(os.getenv(var))
        st.markdown(f"{'✅' if present else '❌'} `{var}`")

topic = st.text_input(
    "Topic",
    placeholder="e.g. Why every data scientist should learn LangGraph",
)
generate = st.button("Generate post", type="primary", use_container_width=True)

if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    app = build_app()
    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    history_container = st.container()
    final_state: dict = {}
    seen_attempts = set()

    with st.status("Starting the writer/reviewer loop...", expanded=True) as status:
        try:
            for update in app.stream(initial_state, stream_mode="updates"):
                node_name, node_output = list(update.items())[0]
                final_state.update(node_output)

                if node_name == "writer":
                    attempt = node_output.get("attempt")
                    last_msg = node_output["messages"][-1]
                    if getattr(last_msg, "tool_calls", None):
                        status.update(
                            label=f"Attempt {attempt}: writer is searching the web...")
                    else:
                        status.update(
                            label=f"Attempt {attempt}: writer is drafting...")

                elif node_name == "tools":
                    status.update(
                        label="Search results retrieved, resuming writing...")

                elif node_name == "extract_draft":
                    status.update(label="Draft ready, sending to reviewer...")

                elif node_name == "reviewer":
                    attempt = final_state.get("attempt", 0)
                    approved = node_output.get("is_approved")
                    feedback = node_output.get("review_feedback", "")

                    if attempt not in seen_attempts:
                        seen_attempts.add(attempt)
                        with history_container.expander(
                            f"Attempt {attempt}", expanded=not approved
                        ):
                            st.markdown(
                                f"**Draft:**\n\n{final_state.get('draft', '')}")
                            verdict_class = "verdict-approved" if approved else "verdict-rejected"
                            verdict_text = "✅ APPROVED" if approved else "❌ REJECTED"
                            st.markdown(
                                f'<span class="{verdict_class}">{verdict_text}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Reviewer feedback:** {feedback}")

                    if approved:
                        status.update(label="Post approved!", state="complete")
                    elif attempt >= 3:
                        status.update(
                            label="Max attempts reached — using last draft.", state="complete")
                    else:
                        status.update(
                            label=f"Rejected — revising for attempt {attempt + 1}...")

        except Exception as e:
            status.update(label="Something went wrong.", state="error")
            st.error(f"⚠️ {e}")
            st.stop()

    st.divider()
    st.subheader("📌 Final result")

    if final_state.get("is_approved"):
        st.success(f"Approved after {final_state.get('attempt')} attempt(s).")
    else:
        st.warning(
            f"Reached max attempts ({final_state.get('attempt')}) — showing the last draft.")

    st.text_area("Final post", value=final_state.get("draft", ""), height=280)
    st.download_button(
        "⬇️ Download as .txt",
        data=final_state.get("draft", ""),
        file_name="linkedin_post.txt",
        mime="text/plain",
        use_container_width=True,
    )
