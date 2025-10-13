import os
import json
import networkx as nx
from typing import TypedDict, Optional, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from pkg import image_upload_display, memory_kg

class AgentState(TypedDict):
    image_path: Optional[str]
    messages: Annotated[list[BaseMessage], operator.add]
    knowledge_graph: nx.DiGraph


def upload_node(state: AgentState) -> AgentState:
    """Upload/select an image."""
    image_path = image_upload_display.run()
    if image_path:
        state["image_path"] = image_path
    return state


def chat_node(state: AgentState) -> AgentState:
    """Chat with GPT-4o about the selected image."""
    image_path = state.get("image_path")
    if not image_path:
        return state

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    sys_msg = SystemMessage(content="You are an assistant that analyzes images and answers questions about them.")

    messages = [sys_msg] + state.get("messages", [])

    response = llm.invoke(messages + [HumanMessage(content=f"[Image: {image_path}]")])

    state["messages"].append(response)

    return state

def update_graph_node(state: AgentState) -> AgentState:
    """Update knowledge graph + save conversation to JSON."""
    G = nx.DiGraph()

    for i, msg in enumerate(state["messages"]):
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"

        content = msg.content
        node_id = f"{role}_{i}"
        G.add_node(node_id, label=content)

        if i > 0:
            prev_msg = state["messages"][i - 1]
            prev_role = "user" if isinstance(prev_msg, HumanMessage) else "assistant"
            prev_node = f"{prev_role}_{i-1}"
            G.add_edge(prev_node, node_id, label="next")

    conv = [{"role": msg.type, "content": msg.content} for msg in state["messages"]]
    with open("conversation.json", "w", encoding="utf-8") as f:
        json.dump(conv, f, indent=2, ensure_ascii=False)

    state["knowledge_graph"] = G
    return state

def visualize_graph_node(state: AgentState) -> AgentState:
    state["knowledge_graph"] = state.get("knowledge_graph", nx.DiGraph())
    return state

def build_agent():
    builder = StateGraph(AgentState)
    builder.add_node("update_graph", update_graph_node)
    builder.add_node("visualize", visualize_graph_node)

    builder.set_entry_point("update_graph")
    builder.add_edge("update_graph", "visualize")
    builder.add_edge("visualize", END)

    return builder.compile()