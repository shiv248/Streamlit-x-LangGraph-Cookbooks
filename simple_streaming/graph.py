from typing import Annotated, TypedDict, Any, Optional

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI

class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    callbacks: Optional[list[Any]]

graph = StateGraph(GraphsState)

def _call_model(state: GraphsState):
    messages = state["messages"]
    callables = state["callbacks"]
    # can swap this llm out with any other LangChain ChatLLM
    # and use the `callbacks` attribute to handle output events
    llm = ChatOpenAI(
        temperature=0.0,
        streaming=True,
        callbacks=callables
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
    return graph_runnable.invoke({"messages": st_messages, "callbacks": callables})
