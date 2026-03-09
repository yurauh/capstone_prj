import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from rag_chatbot import (
    CHROMA_DIR,
    COLLECTION_NAME,
    LOG_FILE_PATH,
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    DocumentChunk,
    build_document_chunks_from_sources,
    create_chroma_client,
    ensure_api_key_is_set,
    generate_answer_from_context,
    populate_vector_store,
    query_vector_store,
)


MAX_AGENT_STEPS: int = 4
DEFAULT_TOP_K: int = 8
DEFAULT_EVALUATION_PATH: str = "evaluation_dataset.json"


@dataclass(frozen=True)
class ToolResult:
    name: str
    input_value: str
    output_value: str
    success: bool


@dataclass(frozen=True)
class ReflectionResult:
    is_sufficient: bool
    critique: str
    next_action_hint: str


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    relevance: float
    clarity: float
    overall: float


@dataclass(frozen=True)
class EvaluationRow:
    question: str
    answer: str
    expected_answer: str
    metrics: EvaluationMetrics


def call_ollama(prompt: str, temperature: float = 0.2) -> str:
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data: Dict[str, Any] = response.json()
    return str(data.get("response", "")).strip()


def parse_first_json_object(raw_text: str) -> Dict[str, Any]:
    text: str = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = text.replace("```", "").strip()
    start: int = text.find("{")
    end: int = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate: str = text[start : end + 1]
    try:
        parsed: Dict[str, Any] = json.loads(candidate)
        return parsed
    except json.JSONDecodeError:
        return {}


def normalize_text(text: str) -> str:
    lowered: str = text.lower()
    cleaned: str = re.sub(r"[^a-z0-9\s]", " ", lowered)
    squashed: str = re.sub(r"\s+", " ", cleaned).strip()
    return squashed


def tokenize_for_overlap(text: str) -> List[str]:
    tokens: List[str] = normalize_text(text).split(" ")
    filtered_tokens: List[str] = [token for token in tokens if token and len(token) > 2]
    return filtered_tokens


def compute_keyword_coverage(answer: str, expected_answer: str) -> float:
    expected_tokens: List[str] = tokenize_for_overlap(expected_answer)
    if not expected_tokens:
        return 0.0
    answer_tokens: set[str] = set(tokenize_for_overlap(answer))
    matched_count: int = 0
    for token in set(expected_tokens):
        if token in answer_tokens:
            matched_count += 1
    return matched_count / len(set(expected_tokens))


def compute_relevance_from_question(question: str, answer: str) -> float:
    question_tokens: set[str] = set(tokenize_for_overlap(question))
    if not question_tokens:
        return 0.0
    answer_tokens: set[str] = set(tokenize_for_overlap(answer))
    intersection_count: int = len(question_tokens.intersection(answer_tokens))
    return intersection_count / len(question_tokens)


def compute_clarity_score(answer: str) -> float:
    stripped: str = answer.strip()
    if not stripped:
        return 0.0
    sentences: List[str] = [part.strip() for part in re.split(r"[.!?]+", stripped) if part.strip()]
    if not sentences:
        return 0.0
    sentence_lengths: List[int] = [len(tokenize_for_overlap(sentence)) for sentence in sentences]
    average_length: float = sum(sentence_lengths) / max(1, len(sentence_lengths))
    has_structure_bonus: float = 0.1 if ("\n-" in answer or "\n1." in answer) else 0.0
    if average_length < 5:
        base: float = 0.4
    elif average_length <= 24:
        base = 0.9
    elif average_length <= 35:
        base = 0.7
    else:
        base = 0.5
    final_score: float = min(1.0, base + has_structure_bonus)
    return final_score


def compute_overall_metrics(question: str, answer: str, expected_answer: str) -> EvaluationMetrics:
    accuracy: float = compute_keyword_coverage(answer=answer, expected_answer=expected_answer)
    relevance: float = compute_relevance_from_question(question=question, answer=answer)
    clarity: float = compute_clarity_score(answer=answer)
    overall: float = (accuracy * 0.45) + (relevance * 0.35) + (clarity * 0.20)
    return EvaluationMetrics(accuracy=accuracy, relevance=relevance, clarity=clarity, overall=overall)


def format_context_chunks(chunks: List[DocumentChunk], limit: int = 4) -> str:
    if not chunks:
        return "No chunks were retrieved."
    selected_chunks: List[DocumentChunk] = chunks[:limit]
    formatted_sections: List[str] = []
    for chunk in selected_chunks:
        source: str = str(chunk.metadata.get("source", "unknown"))
        file_path: str = str(chunk.metadata.get("file_path", "unknown"))
        index: str = str(chunk.metadata.get("chunk_index", "n/a"))
        preview: str = chunk.text[:450].replace("\n", " ").strip()
        section: str = f"- [{source}] {file_path} (chunk {index}) :: {preview}"
        formatted_sections.append(section)
    return "\n".join(formatted_sections)


def run_tool_search_knowledge(chroma_client: Any, query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[ToolResult, List[DocumentChunk]]:
    retrieved_chunks: List[DocumentChunk] = query_vector_store(
        chroma_client=chroma_client,
        query=query,
        top_k=top_k,
        collection_name=COLLECTION_NAME,
    )
    output: str = format_context_chunks(chunks=retrieved_chunks, limit=4)
    tool_result: ToolResult = ToolResult(
        name="search_knowledge_base",
        input_value=query,
        output_value=output,
        success=True,
    )
    return tool_result, retrieved_chunks


def run_tool_inspect_project_file(file_path: str) -> ToolResult:
    workspace_root: Path = Path(__file__).resolve().parent
    candidate_path: Path = (workspace_root / file_path).resolve()
    if workspace_root not in candidate_path.parents and candidate_path != workspace_root:
        return ToolResult(
            name="inspect_project_file",
            input_value=file_path,
            output_value="Blocked path. Only files inside this project are allowed.",
            success=False,
        )
    if not candidate_path.exists():
        return ToolResult(
            name="inspect_project_file",
            input_value=file_path,
            output_value="File does not exist.",
            success=False,
        )
    try:
        content: str = candidate_path.read_text(encoding="utf-8")
    except OSError as err:
        return ToolResult(
            name="inspect_project_file",
            input_value=file_path,
            output_value=f"Failed to read file: {err}",
            success=False,
        )
    preview: str = content[:1200]
    output: str = f"File: {candidate_path}\nPreview:\n{preview}"
    return ToolResult(
        name="inspect_project_file",
        input_value=file_path,
        output_value=output,
        success=True,
    )


def run_tool_inspect_chat_logs(max_lines: int = 30) -> ToolResult:
    if not os.path.exists(LOG_FILE_PATH):
        return ToolResult(
            name="inspect_chat_logs",
            input_value=str(max_lines),
            output_value="No chat log file found yet.",
            success=False,
        )
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as log_file:
            lines: List[str] = log_file.readlines()
    except OSError as err:
        return ToolResult(
            name="inspect_chat_logs",
            input_value=str(max_lines),
            output_value=f"Failed to read logs: {err}",
            success=False,
        )
    tail_lines: List[str] = lines[-max_lines:]
    return ToolResult(
        name="inspect_chat_logs",
        input_value=str(max_lines),
        output_value="".join(tail_lines).strip(),
        success=True,
    )


def build_planner_prompt(
    question: str,
    executed_tools: List[ToolResult],
    reflection_notes: List[str],
) -> str:
    tool_history: str = "\n".join(
        [
            f"{index + 1}. {item.name}({item.input_value}) -> {item.output_value[:360]}"
            for index, item in enumerate(executed_tools)
        ]
    )
    reflections: str = "\n".join([f"{index + 1}. {note}" for index, note in enumerate(reflection_notes)])
    if not tool_history:
        tool_history = "No tools used yet."
    if not reflections:
        reflections = "No reflections yet."
    prompt: str = (
        "You are an autonomous planning assistant inside a RAG system.\n"
        "You can choose exactly one next action.\n"
        "Available tools:\n"
        "1) search_knowledge_base: use to retrieve relevant chunks for a query.\n"
        "2) inspect_project_file: use to inspect one local file path in this project.\n"
        "3) inspect_chat_logs: use to inspect recent chat logs.\n"
        "4) finish: use when enough information exists to answer.\n\n"
        f"User question:\n{question}\n\n"
        f"Tool history:\n{tool_history}\n\n"
        f"Reflection notes:\n{reflections}\n\n"
        "Respond as strict JSON with keys: thought, tool, tool_input, reason.\n"
        "Example: {\"thought\":\"...\",\"tool\":\"search_knowledge_base\",\"tool_input\":\"...\",\"reason\":\"...\"}\n"
        "Do not include markdown."
    )
    return prompt


def choose_next_action(
    question: str,
    executed_tools: List[ToolResult],
    reflection_notes: List[str],
) -> Dict[str, str]:
    prompt: str = build_planner_prompt(
        question=question,
        executed_tools=executed_tools,
        reflection_notes=reflection_notes,
    )
    raw_response: str = call_ollama(prompt=prompt, temperature=0.1)
    parsed: Dict[str, Any] = parse_first_json_object(raw_text=raw_response)
    tool_name: str = str(parsed.get("tool", "search_knowledge_base")).strip()
    tool_input: str = str(parsed.get("tool_input", question)).strip()
    thought: str = str(parsed.get("thought", "Plan the next best action.")).strip()
    reason: str = str(parsed.get("reason", "Use the tool to improve evidence quality.")).strip()
    if tool_name not in {"search_knowledge_base", "inspect_project_file", "inspect_chat_logs", "finish"}:
        tool_name = "search_knowledge_base"
    if not tool_input:
        tool_input = question
    return {"tool": tool_name, "tool_input": tool_input, "thought": thought, "reason": reason}


def build_reflection_prompt(question: str, tool_result: ToolResult, context_count: int) -> str:
    prompt: str = (
        "You are a strict self-reflection assistant for an autonomous RAG agent.\n"
        "Evaluate if the latest tool output is sufficient to answer the user question reliably.\n"
        f"Question: {question}\n"
        f"Tool used: {tool_result.name}\n"
        f"Tool input: {tool_result.input_value}\n"
        f"Tool output:\n{tool_result.output_value[:1400]}\n"
        f"Retrieved chunk count currently available: {context_count}\n\n"
        "Respond as strict JSON with keys: is_sufficient (true/false), critique, next_action_hint.\n"
        "Do not include markdown."
    )
    return prompt


def reflect_on_action(question: str, tool_result: ToolResult, context_count: int) -> ReflectionResult:
    prompt: str = build_reflection_prompt(
        question=question,
        tool_result=tool_result,
        context_count=context_count,
    )
    raw_response: str = call_ollama(prompt=prompt, temperature=0.1)
    parsed: Dict[str, Any] = parse_first_json_object(raw_text=raw_response)
    is_sufficient_value: bool = bool(parsed.get("is_sufficient", False))
    critique: str = str(parsed.get("critique", "Need stronger or more direct evidence from sources.")).strip()
    next_action_hint: str = str(parsed.get("next_action_hint", "Search again with a narrower query.")).strip()
    return ReflectionResult(
        is_sufficient=is_sufficient_value,
        critique=critique,
        next_action_hint=next_action_hint,
    )


def generate_grounded_answer(question: str, context_chunks: List[DocumentChunk]) -> str:
    if not context_chunks:
        return "I do not have enough reliable context to answer this question."
    initial_answer: str = generate_answer_from_context(question=question, context_chunks=context_chunks)
    evidence_list: str = format_context_chunks(chunks=context_chunks, limit=3)
    revision_prompt: str = (
        "Revise the answer to be concise, factual, and grounded in evidence.\n"
        "Add a short evidence section with bullet points citing source file paths.\n"
        f"Question:\n{question}\n\n"
        f"Draft answer:\n{initial_answer}\n\n"
        f"Evidence snippets:\n{evidence_list}\n\n"
        "Output only the final revised answer."
    )
    revised_answer: str = call_ollama(prompt=revision_prompt, temperature=0.1)
    return revised_answer


def execute_planned_tool(action: Dict[str, str], chroma_client: Any) -> Tuple[ToolResult, List[DocumentChunk]]:
    tool_name: str = action["tool"]
    tool_input: str = action["tool_input"]
    if tool_name == "search_knowledge_base":
        return run_tool_search_knowledge(chroma_client=chroma_client, query=tool_input, top_k=DEFAULT_TOP_K)
    if tool_name == "inspect_project_file":
        file_result: ToolResult = run_tool_inspect_project_file(file_path=tool_input)
        return file_result, []
    if tool_name == "inspect_chat_logs":
        log_result: ToolResult = run_tool_inspect_chat_logs(max_lines=30)
        return log_result, []
    finish_result: ToolResult = ToolResult(
        name="finish",
        input_value=tool_input,
        output_value="Planner chose to finish with current evidence.",
        success=True,
    )
    return finish_result, []


def run_agentic_answer(chroma_client: Any, question: str, verbose: bool = True) -> Dict[str, Any]:
    executed_tools: List[ToolResult] = []
    reflection_notes: List[str] = []
    best_context_chunks: List[DocumentChunk] = []
    for step in range(MAX_AGENT_STEPS):
        action: Dict[str, str] = choose_next_action(
            question=question,
            executed_tools=executed_tools,
            reflection_notes=reflection_notes,
        )
        if verbose:
            print(f"\n[Step {step + 1}] Thought: {action['thought']}")
            print(f"[Step {step + 1}] Action: {action['tool']}({action['tool_input']})")
            print(f"[Step {step + 1}] Reason: {action['reason']}")
        tool_result, context_chunks = execute_planned_tool(action=action, chroma_client=chroma_client)
        executed_tools.append(tool_result)
        if context_chunks:
            best_context_chunks = context_chunks
        if verbose:
            print(f"[Step {step + 1}] Tool output preview: {tool_result.output_value[:280]}")
        if tool_result.name == "finish":
            break
        reflection_result: ReflectionResult = reflect_on_action(
            question=question,
            tool_result=tool_result,
            context_count=len(best_context_chunks),
        )
        reflection_notes.append(reflection_result.critique)
        if verbose:
            print(f"[Step {step + 1}] Reflection: {reflection_result.critique}")
            print(f"[Step {step + 1}] Next hint: {reflection_result.next_action_hint}")
        if reflection_result.is_sufficient and best_context_chunks:
            break
    final_answer: str = generate_grounded_answer(question=question, context_chunks=best_context_chunks)
    if verbose:
        print("\nFinal answer:")
        print(final_answer)
    return {
        "question": question,
        "final_answer": final_answer,
        "executed_tools": executed_tools,
        "reflection_notes": reflection_notes,
        "context_chunks": best_context_chunks,
    }


def append_agent_log(question: str, final_answer: str, tool_results: List[ToolResult]) -> None:
    tool_summaries: str = "; ".join([f"{tool.name}({tool.input_value})" for tool in tool_results])
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
            log_file.write("=== Agent Entry ===\n")
            log_file.write(f"Question: {question}\n")
            log_file.write(f"Tools: {tool_summaries}\n")
            log_file.write(f"Answer: {final_answer}\n\n")
    except OSError as err:
        print(f"Warning: failed to write agent log entry: {err}")


def load_sources_from_data_dirs() -> Tuple[List[str], List[str]]:
    pdf_dir: str = os.path.join("data", "pdfs")
    audio_dir: str = os.path.join("data", "audio")
    pdf_paths: List[str] = []
    audio_paths: List[str] = []
    if os.path.isdir(pdf_dir):
        for name in os.listdir(pdf_dir):
            if name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(pdf_dir, name))
    if os.path.isdir(audio_dir):
        for name in os.listdir(audio_dir):
            lower_name: str = name.lower()
            if lower_name.endswith(".wav") or lower_name.endswith(".mp3") or lower_name.endswith(".m4a"):
                audio_paths.append(os.path.join(audio_dir, name))
    return pdf_paths, audio_paths


def execute_build_index() -> None:
    ensure_api_key_is_set()
    pdf_paths, audio_paths = load_sources_from_data_dirs()
    if not pdf_paths and not audio_paths:
        raise RuntimeError(
            "No PDF or audio files found. Put files into data/pdfs and data/audio first."
        )
    print(f"Preparing chunks from {len(pdf_paths)} PDF(s) and {len(audio_paths)} audio file(s)...")
    chunks: List[DocumentChunk] = build_document_chunks_from_sources(
        pdf_paths=pdf_paths,
        audio_paths=audio_paths,
    )
    chroma_client: Any = create_chroma_client(persist_directory=CHROMA_DIR)
    populate_vector_store(
        chroma_client=chroma_client,
        chunks=chunks,
        collection_name=COLLECTION_NAME,
    )
    print("Index build complete.")


def execute_chat_mode() -> None:
    ensure_api_key_is_set()
    chroma_client: Any = create_chroma_client(persist_directory=CHROMA_DIR)
    print("Agentic RAG is ready. Type 'exit' to quit.")
    while True:
        question: str = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not question:
            print("Please enter a question.")
            continue
        try:
            result: Dict[str, Any] = run_agentic_answer(chroma_client=chroma_client, question=question, verbose=True)
            append_agent_log(
                question=question,
                final_answer=str(result["final_answer"]),
                tool_results=list(result["executed_tools"]),
            )
            metrics: EvaluationMetrics = compute_overall_metrics(
                question=question,
                answer=str(result["final_answer"]),
                expected_answer=question,
            )
            print(
                f"\nSelf-eval -> accuracy={metrics.accuracy:.2f}, "
                f"relevance={metrics.relevance:.2f}, clarity={metrics.clarity:.2f}, "
                f"overall={metrics.overall:.2f}"
            )
        except Exception as err:
            print(f"Error: {err}")


def load_evaluation_dataset(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data: Any = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    rows: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question: str = str(item.get("question", "")).strip()
        expected_answer: str = str(item.get("expected_answer", "")).strip()
        if question and expected_answer:
            rows.append({"question": question, "expected_answer": expected_answer})
    if not rows:
        raise ValueError("Evaluation dataset has no valid question/expected_answer entries.")
    return rows


def execute_evaluation_mode(dataset_path: str) -> None:
    ensure_api_key_is_set()
    chroma_client: Any = create_chroma_client(persist_directory=CHROMA_DIR)
    rows: List[Dict[str, str]] = load_evaluation_dataset(path=dataset_path)
    detailed_results: List[EvaluationRow] = []
    print(f"Running evaluation on {len(rows)} questions...")
    for index, row in enumerate(rows):
        question: str = row["question"]
        expected_answer: str = row["expected_answer"]
        print(f"\n[{index + 1}/{len(rows)}] Q: {question}")
        result: Dict[str, Any] = run_agentic_answer(chroma_client=chroma_client, question=question, verbose=False)
        answer: str = str(result["final_answer"])
        metrics: EvaluationMetrics = compute_overall_metrics(
            question=question,
            answer=answer,
            expected_answer=expected_answer,
        )
        detailed_results.append(
            EvaluationRow(
                question=question,
                answer=answer,
                expected_answer=expected_answer,
                metrics=metrics,
            )
        )
        print(
            f"accuracy={metrics.accuracy:.2f}, relevance={metrics.relevance:.2f}, "
            f"clarity={metrics.clarity:.2f}, overall={metrics.overall:.2f}"
        )
    if not detailed_results:
        print("No results.")
        return
    mean_accuracy: float = sum([item.metrics.accuracy for item in detailed_results]) / len(detailed_results)
    mean_relevance: float = sum([item.metrics.relevance for item in detailed_results]) / len(detailed_results)
    mean_clarity: float = sum([item.metrics.clarity for item in detailed_results]) / len(detailed_results)
    mean_overall: float = sum([item.metrics.overall for item in detailed_results]) / len(detailed_results)
    print("\n=== Evaluation Summary ===")
    print(f"Mean accuracy : {mean_accuracy:.3f}")
    print(f"Mean relevance: {mean_relevance:.3f}")
    print(f"Mean clarity  : {mean_clarity:.3f}")
    print(f"Mean overall  : {mean_overall:.3f}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic RAG system with reflection and evaluation.")
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=["chat", "build_index", "evaluate"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=DEFAULT_EVALUATION_PATH,
        help="Path to evaluation dataset JSON file for --mode evaluate.",
    )
    return parser


def main() -> None:
    parser: argparse.ArgumentParser = build_argument_parser()
    args: argparse.Namespace = parser.parse_args()
    if args.mode == "build_index":
        execute_build_index()
        return
    if args.mode == "evaluate":
        execute_evaluation_mode(dataset_path=args.eval_file)
        return
    execute_chat_mode()


if __name__ == "__main__":
    main()
