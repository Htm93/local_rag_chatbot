"""
LangGraph RAG pipeline, extended with short-term conversation memory.

AgentState now carries `chat_history`: a list of (user, ai) tuples. The
caller (FastAPI layer) is responsible for persisting this per chat session
and trimming it to the last MAX_HISTORY_TURNS turns before each call -
run_agent() also enforces the cap defensively.
"""

from typing import List, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import END, StateGraph

from vector_search import retriever

MODEL_NAME = "llama3:8b"
model = OllamaLLM(model=MODEL_NAME)

MAX_HISTORY_TURNS = 5

template = """
You are an expert information synthesizer and researcher.
Answer the user's question based STRICTLY on the provided data below.

LANGUAGE RULE: Always respond in the SAME language the question was asked
in. For example, if the question is in Vietnamese, answer entirely in
Vietnamese; if it's in English, answer entirely in English - regardless of
what language the provided data below is written in.

Conversation history is provided only so you can resolve references such as
"it", "that", or "the one you mentioned" - never treat it as a source of
facts. Facts must come only from the provided data.

IMPORTANT CONSTRAINTS:
- Do not make up any information or use outside knowledge.
- If the provided data does not contain the information needed to answer
  the question, you MUST answer with: "I don't know" (in English) or its
  equivalent in the question's language (e.g. "Tôi không biết" in
  Vietnamese).

Conversation history (oldest to newest):
{history}

Provided data:
{data}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ---- Meta-command handling (translate/shorten/rephrase the last answer) ----
# Keyword-based detection: if the question looks like an instruction to
# transform the *previous* answer rather than a new factual question, we
# skip retrieval entirely and just ask the LLM to transform the existing
# text. This keeps such follow-ups fast and avoids the strict "answer only
# from retrieved data" rule incorrectly producing "I don't know".
META_KEYWORDS = [
    "dịch", "translate", "rephrase", "reword", "paraphrase",
    "tóm tắt", "summarize", "summarise", "shorter", "ngắn hơn",
    "explain simpler", "explain it simpler", "in english", "in vietnamese",
    "viết lại", "rewrite", "simplify", "expand on that", "elaborate",
]

transform_template = """
You are a helpful writing assistant. The user previously received this answer:

{previous_answer}

Now apply the following instruction to that answer ONLY. Do not add new
facts that weren't already in the previous answer - just transform its
wording, language, or length as instructed. If the instruction asks for a
translation, translate fully into the requested language. Otherwise, keep
your response in the same language the instruction itself is written in.

Instruction: {question}
"""
transform_prompt = ChatPromptTemplate.from_template(transform_template)
transform_chain = transform_prompt | model


def is_meta_command(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in META_KEYWORDS)


# ---------------------- LANGGRAPH STATE ----------------------
class AgentState(TypedDict):
    """Memory of the LangGraph run. Data flows between nodes through this."""

    question: str
    data: List[Document]
    answer: str
    loop_step: int  # count loops to prevent infinite retry
    chat_history: List[Tuple[str, str]]  # last N (user, ai) turns


def _format_docs(docs: List[Document]) -> str:
    """Turn retrieved Document objects into clean, readable text for the
    prompt. Passing raw Document objects directly (their Python repr,
    metadata dict and all) confuses smaller local models."""
    if not docs:
        return "(no matching data found)"
    parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown source")
        parts.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "(no previous turns)"
    trimmed = history[-MAX_HISTORY_TURNS:]
    lines = [f"User: {u}\nAI: {a}" for u, a in trimmed]
    return "\n".join(lines)


def transform_node(state: AgentState):
    """Handle meta-commands (translate/shorten/rephrase) on the previous
    answer directly, without touching the vector store."""
    print("\n[NODE: TRANSFORM] Applying instruction to previous answer...")
    question = state["question"]
    history = state.get("chat_history", [])
    previous_answer = history[-1][1] if history else "(no previous answer available)"

    result = transform_chain.invoke({"previous_answer": previous_answer, "question": question})
    return {"answer": result, "data": [], "loop_step": state.get("loop_step", 0) + 1}


def retrieve_node(state: AgentState):
    """Node 1: look up data in the vector DB."""
    print("\n[NODE: RETRIEVER] Looking for data in ChromaDB...")
    question = state["question"]
    data = retriever.invoke(question)
    return {"data": data, "loop_step": state.get("loop_step", 0) + 1}


def generate_node(state: AgentState):
    """Node 2: LLM answers using retrieved data + recent chat history."""
    print(f"[NODE: GENERATOR] LLM ({MODEL_NAME}) processing answer...")
    question = state["question"]
    data = state["data"]
    history_str = _format_history(state.get("chat_history", []))

    result = chain.invoke(
        {"data": _format_docs(data), "question": question, "history": history_str}
    )
    return {"answer": result}


def evaluate_answer_edge(state: AgentState):
    """Self-check: retry retrieval once if the model says it doesn't know."""
    print("[EDGE: FACT CHECKER] Checking answer quality...")
    answer = state["answer"].lower()
    step = state.get("loop_step", 0)

    if step >= 2:
        print("-> [WARNING] Maximum repetition reached, confirming answer.")
        return "pass"

    if (
        "don't know" in answer
        or "do not know" in answer
        or "not mention" in answer
        or "không biết" in answer
        or "không đề cập" in answer
    ):
        print("-> [ERROR] LLM lacks information. Retrying retrieval.")
        return "retry"

    print("[SUCCESS] Good answer, ready to output.")
    return "pass"


# ---------------------- GRAPH WIRING ----------------------
def route_entry(state: AgentState) -> str:
    """Send meta-commands (translate/shorten/rephrase) straight to the
    transform node; everything else goes through normal retrieval."""
    if state.get("chat_history") and is_meta_command(state["question"]):
        return "transform"
    return "retrieve"


workflow = StateGraph(AgentState)
workflow.add_node("transform", transform_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generator", generate_node)
workflow.set_conditional_entry_point(route_entry, {"transform": "transform", "retrieve": "retrieve"})
workflow.add_edge("transform", END)
workflow.add_edge("retrieve", "generator")
workflow.add_conditional_edges(
    "generator",
    evaluate_answer_edge,
    {"retry": "retrieve", "pass": END},
)

app = workflow.compile()


def run_agent(question: str, chat_history: List[Tuple[str, str]] | None = None) -> str:
    """Convenience wrapper used by the FastAPI backend."""
    inputs = {
        "question": question,
        "loop_step": 0,
        "chat_history": (chat_history or [])[-MAX_HISTORY_TURNS:],
    }

    final_state = None
    for output_state in app.stream(inputs):
        for _, state_value in output_state.items():
            final_state = state_value

    return final_state.get("answer", "Error: No answer") if final_state else "Error: No answer"


if __name__ == "__main__":
    # Same CLI loop as the original script, now with memory, for quick testing.
    print("\n[SYSTEM] Starting LangGraph agent (CLI mode, with memory)...")
    history: List[Tuple[str, str]] = []

    while True:
        print("\n" + "=" * 40)
        question = input("Ask your question (q to quit): ")
        if question.lower() == "q":
            print("Goodbye!")
            break

        answer = run_agent(question, chat_history=history)
        history.append((question, answer))
        history = history[-MAX_HISTORY_TURNS:]

        print("\n[FINAL ANSWER]:")
        print(answer)