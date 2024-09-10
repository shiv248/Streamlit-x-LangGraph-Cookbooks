import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb

load_dotenv()

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### StreamlitCallBackHandler Full Implementation")

"""
In this example, we're going to be using [`StreamlitCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_community.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html) 
within [_LangGraph_](https://langchain-ai.github.io/langgraph/) by leveraging callbacks in our 
graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).
"""

if not os.getenv('OPENAI_API_KEY'):
    st.sidebar.header("OPENAI Setup")
    api_key = st.sidebar.text_input("API_Key", type="password")
    os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

for msg in st.session_state.messages:
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        st_callback = get_streamlit_cb(st.empty())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        last_msg = response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=last_msg))
        msg_placeholder.write(last_msg)
