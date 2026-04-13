"""
LangGraph ReAct workflow for the RAG agent.

Adapted from DeepXiv's graph.py with two changes:
  - State keys updated for documents (docs, chunk_cache, etc.)
  - System prompt and tool executor use RAG-specific implementations
"""

import time
import json
from typing import Dict, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from openai import OpenAI

from .state import AgentState
from .tools import get_tools_definition, ToolExecutor, format_doc_context
from .prompts import get_system_prompt


def call_llm(
    messages: List[Dict],
    client: OpenAI,
    model_name: str = "gpt-4",
    tools: Optional[List] = None,
    max_tries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    stream: bool = False,
    print_process: bool = False,
) -> tuple:
    """
    Call LLM with retry and exponential back-off.
    Returns (content: str, tool_calls: list | None).
    """
    import random

    for attempt in range(max_tries):
        try:
            params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            if stream:
                response_stream = client.chat.completions.create(**params)
                content = ""
                tool_calls_dict: Dict = {}

                for chunk in response_stream:
                    if not hasattr(chunk.choices[0], "delta"):
                        continue
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content
                        if print_process:
                            print(delta.content, end="", flush=True)
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {
                                    "id": tc.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if hasattr(tc, "function"):
                                if tc.function.name:
                                    tool_calls_dict[idx]["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

                if print_process and content:
                    print()

                tool_calls = (
                    [tool_calls_dict[i] for i in sorted(tool_calls_dict)] if tool_calls_dict else None
                )
                if content.strip():
                    return content.strip(), tool_calls
                if tool_calls:
                    return "", tool_calls
            else:
                response = client.chat.completions.create(**params)
                message = response.choices[0].message
                content = message.content or ""
                tool_calls = None
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                if content.strip():
                    if print_process:
                        print(f"[LLM] {content[:200]}...")
                    return content.strip(), tool_calls
                if tool_calls:
                    return "", tool_calls

        except Exception as e:
            if print_process:
                print(f"[LLM] Attempt {attempt + 1} failed: {e}")

        if attempt < max_tries - 1:
            sleep_time = min(2 ** attempt + random.uniform(0, 1), 10)
            time.sleep(sleep_time)

    return "LLM 服务暂时不可用，请稍后重试。", None


# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------

def planning_node(state: AgentState, config: RunnableConfig) -> AgentState:
    cfg = config.get("configurable", {})
    max_time = cfg.get("max_time_seconds", 600)
    print_process = cfg.get("print_process", False)

    # Timeout guard
    if time.time() - state["start_time"] > max_time:
        return {
            "prediction": "超时：未能在限定时间内找到答案。",
            "termination": "timeout",
            "status": state["status"] + ["timeout"],
        }

    if state["num_llm_calls_available"] <= 0:
        return {
            "prediction": "已达最大调用次数，无法继续检索。",
            "termination": "exceeded_calls",
            "status": state["status"] + ["exceeded_calls"],
        }

    messages = state.get("messages", [])

    # Initialize conversation on first round
    if not messages:
        doc_context = format_doc_context(state["docs"])
        current_date = time.strftime("%Y-%m-%d")
        system_content = get_system_prompt(doc_context, current_date)
        initial_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": state["question"]},
        ]
        messages_to_add = initial_messages
        all_messages = initial_messages
    else:
        messages_to_add = []
        all_messages = messages

    tools = get_tools_definition()
    client = cfg.get("client")
    model_name = cfg.get("model_name", "gpt-4")

    content, tool_calls = call_llm(
        messages=all_messages,
        client=client,
        model_name=model_name,
        tools=tools,
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=cfg.get("temperature", 0.7),
        stream=cfg.get("stream", False),
        print_process=print_process,
    )

    if print_process:
        print(f"\n{'='*60}")
        print(f"[Round {state['round'] + 1}] tool_calls={len(tool_calls) if tool_calls else 0}")
        print(f"{'='*60}")

    assistant_msg = {"role": "assistant", "content": content.strip()}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    messages_to_add.append(assistant_msg)

    new_status = state["status"].copy()
    if tool_calls:
        new_status.append("tool_call")
    elif "<answer>" in content and "</answer>" in content:
        new_status.append("answer")
    elif state["round"] >= 1 and len(content.strip()) > 50:
        new_status.append("answer")
    else:
        new_status.append("continue")

    return {
        "messages": messages_to_add,
        "response": content.strip(),
        "status": new_status,
        "round": state["round"] + 1,
        "num_llm_calls_available": state["num_llm_calls_available"] - 1,
    }


def tool_call_node(state: AgentState, config: RunnableConfig) -> AgentState:
    cfg = config.get("configurable", {})
    print_process = cfg.get("print_process", False)
    tool_executor: ToolExecutor = cfg.get("tool_executor")

    # Find the most recent assistant message with tool_calls
    tool_calls = None
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            break

    if not tool_calls:
        return {
            "messages": [{"role": "tool", "content": "错误：未找到工具调用请求。"}],
            "status": state["status"] + ["tool_response"],
        }

    tool_results = []
    for tc in tool_calls:
        tool_id = tc.get("id", "")
        fn_name = tc.get("function", {}).get("name", "")
        fn_args_str = tc.get("function", {}).get("arguments", "{}")

        try:
            fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
        except Exception:
            fn_args = {}

        if print_process:
            print(f"\n[Tool] {fn_name}({fn_args})")

        result = tool_executor.execute_tool_call(fn_name, fn_args, state)

        if print_process:
            print(f"[Tool result] {result[:300]}...")

        tool_results.append(
            {"role": "tool", "tool_call_id": tool_id, "content": result}
        )

    return {
        "messages": tool_results,
        "status": state["status"] + ["tool_response"],
    }


def check_limits_node(state: AgentState, config: RunnableConfig) -> AgentState:
    cfg = config.get("configurable", {})
    max_llm_calls = cfg.get("max_llm_calls", 20)
    print_process = cfg.get("print_process", False)

    if state["round"] < max_llm_calls - 2:
        return {"status": state["status"]}

    if print_process:
        print(f"\n[Limits] 接近调用上限，请求最终回答...")

    messages = list(state.get("messages", []))
    limit_msg = (
        "你已接近最大调用次数，请根据目前收集到的所有信息，"
        "立即给出最终回答。请用 <answer></answer> 标签包裹你的回答。"
    )
    messages.append({"role": "user", "content": limit_msg})

    client = cfg.get("client")
    model_name = cfg.get("model_name", "gpt-4")

    content, _ = call_llm(
        messages=messages,
        client=client,
        model_name=model_name,
        tools=None,  # no tools — force a direct answer
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=cfg.get("temperature", 0.7),
        stream=cfg.get("stream", False),
        print_process=print_process,
    )

    if "<answer>" in content and "</answer>" in content:
        prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
        termination = "answer (limit reached)"
    else:
        prediction = content.strip()
        termination = "limit reached"

    return {
        "messages": [
            {"role": "user", "content": limit_msg},
            {"role": "assistant", "content": content.strip()},
        ],
        "response": content.strip(),
        "prediction": prediction,
        "termination": termination,
        "status": state["status"] + ["answer"],
        "round": state["round"] + 1,
        "num_llm_calls_available": state["num_llm_calls_available"] - 1,
    }


def router_node(state: AgentState) -> Literal["tool_call", "check_limits", "answer", "continue"]:
    if not state.get("status"):
        return "continue"
    last = state["status"][-1]
    if last == "tool_call":
        return "tool_call"
    if last in ("answer", "timeout", "exceeded_calls"):
        return "answer"
    return "check_limits"


def finalize_node(state: AgentState) -> AgentState:
    response = state.get("response", "")

    if "<answer>" in response and "</answer>" in response:
        prediction = response.split("<answer>")[1].split("</answer>")[0].strip()
        return {"prediction": prediction, "termination": "answer"}

    # Search through message history
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "<answer>" in content and "</answer>" in content:
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                return {"prediction": prediction, "termination": "answer"}
            if content and len(content.strip()) > 20 and not msg.get("tool_calls"):
                return {"prediction": content.strip(), "termination": "answer (no tags)"}

    return {"prediction": "未能找到答案。", "termination": "no_answer"}


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def create_react_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("planning", planning_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("check_limits", check_limits_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("planning")

    workflow.add_conditional_edges(
        "planning",
        router_node,
        {
            "tool_call": "tool_call",
            "check_limits": "check_limits",
            "answer": "finalize",
            "continue": "check_limits",
        },
    )

    def route_after_limits(state: AgentState) -> Literal["planning", "finalize"]:
        if state.get("status") and state["status"][-1] == "answer":
            return "finalize"
        return "planning"

    workflow.add_conditional_edges(
        "check_limits",
        route_after_limits,
        {"planning": "planning", "finalize": "finalize"},
    )

    workflow.add_edge("tool_call", "check_limits")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def create_initial_state(session_id: str = "", docs: Optional[Dict] = None) -> AgentState:
    return {
        "session_id": session_id,
        "docs": docs or {},
        "messages": [],
        "question": "",
        "response": "",
        "status": [],
        "round": 0,
        "num_llm_calls_available": 20,
        "start_time": time.time(),
        "prediction": "",
        "termination": "",
        "chunk_cache": {},
        "search_results_cache": [],
    }
