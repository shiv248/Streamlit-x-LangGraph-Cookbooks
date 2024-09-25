import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

load_dotenv()

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### Simple Chat Streaming")

# st write magic
"""
In this example, we're going to be creating our own [`BaseCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html) called StreamHandler 
to stream our [_LangGraph_](https://langchain-ai.github.io/langgraph/) invocations and leveraging callbacks in our 
graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).

The BaseCallBackHandler is a [Mixin](https://www.wikiwand.com/en/articles/Mixin) overloader function which we will use
to implement only `on_llm_new_token`, a method that run on every new generation of a token from the ChatLLM model.

--- 
"""

# Check if the API key is available as an environment variable
if not os.getenv('OPENAI_API_KEY'):
    # If not, display a sidebar input for the user to provide the API key
    st.sidebar.header("OPENAI_API_KEY Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = api_key
    # If no key is provided, show an info message and stop further execution and wait till key is entered
    if not api_key:
        st.info("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handles graph events using the callback mechanism
    with st.chat_message("assistant"):
        # create a new container for streaming messages only, and give it context
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        # Add that last message to the st_message_state
        # Streamlit's refresh the message will automatically be visually rendered bc of the msg render for loop above
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))
