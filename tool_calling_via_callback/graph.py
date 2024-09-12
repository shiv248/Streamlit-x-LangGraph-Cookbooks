from typing import Annotated, TypedDict, Any, Optional, Literal

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI

search_DDG = StructuredTool.from_function(
        name="Search",
        func=DuckDuckGoSearchAPIWrapper().run,
        description=f"""
        useful for when you need to answer questions about current events, anything over the web, or more information.
        You should ask targeted questions
        """,
    )

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

@tool
def get_coolest_cities():
    """Get a list of coolest cities."""
    return "nyc, sf"

tools = [get_weather, get_coolest_cities, search_DDG]
tool_node = ToolNode(tools)

class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph = StateGraph(GraphsState)

def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
    ).bind_tools(tools)
    response = llm.invoke(messages)
    return {"messages": [response]}

graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

graph.add_conditional_edges(
    "modelNode",
    should_continue,
)
graph.add_edge("tools", "modelNode")

graph_runnable = graph.compile()

def invoke_our_graph(st_messages, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})
