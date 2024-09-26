from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig
#  https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.manager.adispatch_custom_event.html
from langchain_core.callbacks import adispatch_custom_event
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt  # interrupt for human-in-the-loop intervention

# This state contains a single field "input" which holds the user-provided string or None for graph resume
class State(TypedDict):
    input: str

# we pass RunnableConfig in case graph is running in Python 3.10 or earlier
# https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#:~:text=Note%20on%20Python%20%3C%203.11
async def step_1(state, config: RunnableConfig):
    print("---Step 1---")
    # Dispatch a custom event to notify about the initialization of the input
    # currently we don't do anything with this, but flexibility is there
    await adispatch_custom_event("on_init_input", {"input": state["input"]}, config=config)
    return state

async def step_2(state, config: RunnableConfig):
    print("---Step 2---")
    if len(state['input']) > 5:
        # Dispatch an event indicating the need for user response (input is too long)
        await adispatch_custom_event("on_waiting_user_resp", len(state["input"]), config=config)
        # Raise a NodeInterrupt to stop execution and request a shorter input (<= 5 characters)
        raise NodeInterrupt(f"Please provide an input </= than 5 characters, current is {state['input']}, with length {len(state['input'])}")
    else:
        # Dispatch an event confirming that the input meets the length condition
        await adispatch_custom_event("on_conditional_check", len(state["input"]), config=config)

async def step_3(state, config: RunnableConfig):
    print("---Step 3---")
    # Dispatch an event indicating the graph has completed with the given input and its length
    await adispatch_custom_event("on_complete_graph", {"input": state["input"], "len": len(state["input"])}, config=config)
    return state

# Define graph nodes and edges
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Create a memory saver to store graph states by thread and allow state recovery
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
