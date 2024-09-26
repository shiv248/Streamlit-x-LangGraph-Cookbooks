from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_fireworks import ChatFireworks

class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph = StateGraph(GraphsState)

def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatFireworks(
        model="accounts/fireworks/models/firefunction-v2",
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
