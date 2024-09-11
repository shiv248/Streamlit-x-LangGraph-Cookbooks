import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb

load_dotenv()

st.title("StreamLit 🤝 LangGraph")

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True

# Check if the OpenAI API key is set; if not, prompt the user to enter it
if not os.getenv('OPENAI_API_KEY'):
    st.sidebar.header("OPENAI_API_KEY Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

with st.expander(label="Simple Chat Streaming and Tool Calling", expanded=st.session_state.expander_open):
    """
    In this example, we're going to be creating our own [`BaseCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html) called StreamHandler 
    to stream our [_LangGraph_](https://langchain-ai.github.io/langgraph/) invocations with `token streaming` or `tool calling` and leveraging callbacks in our 
    graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).

    The BaseCallBackHandler is a [Mixin](https://www.wikiwand.com/en/articles/Mixin) overloader function which we will use
    to implement `on_llm_new_token`, a method that run on every new generation of a token from the ChatLLM model, and
    `on_tool_start` a method that runs on every tool call invocation even multiple tool calls.
    """

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

# Display all the previous messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle user input if provided
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))