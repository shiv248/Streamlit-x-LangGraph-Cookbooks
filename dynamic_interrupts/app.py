from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import asyncio

from astream_events_handler import invoke_our_graph

load_dotenv()

st.title("StreamLit 🤝 LangGraph")

if "expander_open" not in st.session_state:
    st.session_state.expander_open = True

if 'graph_resume' not in st.session_state:
    st.session_state.graph_resume = False

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Please provide me with a word smaller then 5 letters?")]


prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False

with st.expander(label="Dynamic Interrupts", expanded=st.session_state.expander_open):
    """
    ---
    """

# Display all the previous messages on streamlit refresh
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Trigger graph if user input
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.container()
        shared_state = {
            "graph_resume": st.session_state.graph_resume
        }
        response = asyncio.run(invoke_our_graph(prompt, placeholder, shared_state))
        if type(response) is dict:
            operation = response["op"]
            if operation == "on_waiting_user_resp":
                st.session_state.messages.append(AIMessage(response["msg"]))
                st.write(response["msg"])
                st.session_state.graph_resume = True
            elif operation == "on_new_graph_msg":
                st.session_state.messages.append(AIMessage(response["msg"]))
                st.write(response["msg"])
                st.session_state.graph_resume = False
            else:
                st.error("Received: " + response)
        else:
            st.error("Received: " + response)