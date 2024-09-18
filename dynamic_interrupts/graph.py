from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt


class State(TypedDict):
    input: str

async def step_1(state, config: RunnableConfig):
    print("---Step 1---")
    await adispatch_custom_event("on_init_input", {"input": state["input"]}, config=config)
    return state

async def step_2(state, config: RunnableConfig):
    print("---Step 2---")
    if len(state['input']) > 5:
        await adispatch_custom_event("on_waiting_user_resp", len(state["input"]), config=config)
        raise NodeInterrupt(f"Please provide an input </= than 5 characters, current is {state['input']}, with length {len(state['input'])}")
    else:
        await adispatch_custom_event("on_conditional_check", len(state["input"]), config=config)

async def step_3(state, config: RunnableConfig):
    print("---Step 3---")
    await adispatch_custom_event("on_complete_graph", {"input": state["input"], "len": len(state["input"])},
                                 config=config)
    return state

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
