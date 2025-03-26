import os

from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_fireworks import ChatFireworks

FW_MODEL = os.getenv("fw_model", "accounts/fireworks/models/llama-v3p3-70b-instruct")

class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph = StateGraph(GraphsState)

def _call_model(state: GraphsState):
    print("USING FIREWORKS MODEL, " + FW_MODEL)
    messages = state["messages"]
    llm = ChatFireworks(
        model=FW_MODEL,
        temperature=0.0,
        streaming=True,
    )
    response = llm.invoke(messages)
    return {"messages": [response]}

graph.add_edge(START, "modelNode")
graph.add_node("modelNode", _call_model)
graph.add_edge("modelNode", END)

graph_runnable = graph.compile()

def invoke_our_graph(st_messages, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})
