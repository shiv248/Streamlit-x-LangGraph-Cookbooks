import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb

load_dotenv()

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### Deleting Example")

"""
This example is how to setup a chat with LLM, being able to delete user messages and corresponding AI responses.
Though the deleting is from the `st state` frontend side and the entire `state_messages` get invoked to our graph. 
There definitely is a better way of approaching this. This is revision 1.

---
"""

# This App completely works for deletion and has the basework for edit
# thought I wonder what is the best UI/UX to approach it
# have not implemented edit in its entirety

if not os.getenv('FIREWORKS_API_KEY'):
    st.sidebar.header("FIREWORKS_API_KEY Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["FIREWORKS_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your FIREWORKS_API_KEY in the sidebar.")
        st.stop()

st.markdown("""
    <style>
    .stButton>button {
        background-color: rgba(0, 123, 255, 0.0); /* Blue background with 50% opacity */
        position: relative; /* Positioning context */
        padding: 0px 0px;
        min-width: 33px;
        min-height: 31px;
        left: -40px; /* Move the button to the left */
        opacity: 0; /* Fully invisible */
    }
    .stButton>button:hover {
        opacity: 1; /* Make visible on hover */
    }
    </style>
""", unsafe_allow_html=True)

# def click_edit(val, msg):
#     # Edits the state message creates a new thread
#     print("editing", val, msg)


def click_delete(val):
    index_to_delete = val - 1
    # Check if the index is valid
    if 0 <= index_to_delete < len(st.session_state.messages):
        # Check the type of the message at the index to delete
        if isinstance(st.session_state.messages[index_to_delete], HumanMessage):
            # Remove the user's message
            del st.session_state.messages[index_to_delete]
            # Check if there's an AI message to delete following the user's message
            if index_to_delete < len(st.session_state.messages) and isinstance(
                    st.session_state.messages[index_to_delete], AIMessage):
                del st.session_state.messages[index_to_delete]

msg_view_container = st.container()

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

with msg_view_container:
    for msg_val, msg in enumerate(st.session_state.messages, start=1):
        if isinstance(msg, AIMessage):
            name = "assistant"
        elif isinstance(msg, HumanMessage):
            name = "user"
        else:
            continue
        with st.container():
            # col1, col2, col3 = st.columns([8, 1, 1])
            col1, col3 = st.columns([9, 1])
            with col1:
                st.chat_message(name).write(msg.content)
            if name == "user":
                # with col2:
                #     with st.chat_message(name, avatar=":material/edit:"):
                #         st.button("", key=msg_val, on_click=click_edit, args=[msg_val, msg.content])
                with col3:
                    with st.chat_message(name, avatar=":material/delete:"):
                        st.button("", key=-msg_val, on_click=click_delete, args=[msg_val])

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.container():
        msg_val = len(st.session_state.messages)
        col1, col3 = st.columns([9, 1])
        with col1:
            st.chat_message("user").write(prompt)
        # with col2:
        #     with st.chat_message("user", avatar=":material/edit:"):
        #         st.button("", key=msg_val, on_click=click_edit, args=[msg_val, prompt])
        with col3:
            with st.chat_message("user", avatar=":material/delete:"):
                st.button("", key=-msg_val, on_click=click_delete, args=[msg_val])

    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))
