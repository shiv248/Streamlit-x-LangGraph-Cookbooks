from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import asyncio

from astream_events_handler import invoke_our_graph  # Utility function to handle events from astream_events from graph

load_dotenv()

st.title("StreamLit 🤝 LangGraph")

# Session state management for expander and graph resume after interrupt
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True  # Initially keep expander open

if 'graph_resume' not in st.session_state:
    st.session_state.graph_resume = False  # Track if the graph should resume from a previous state

# Initialize chat messages in session state
if "messages" not in st.session_state:
    # Set an initial message from the "Ai" to prompt the user
    st.session_state["messages"] = [AIMessage(content="Please provide me with a word smaller then 5 letters?")]

prompt = st.chat_input()

if prompt is not None:
    st.session_state.expander_open = False  # Close expander when user enters a prompt

with st.expander(label="Dynamic Interrupts", expanded=st.session_state.expander_open):
    """
    This example will highlight the usage of `NodeInterrupt`s and `adispatch_custom_event` 
    to achieve a custom Human in the Loop experience, 
    that will dynamically ask the user for a new response based on the user's input.
    ---
    """

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Trigger graph interaction if there's a new user input (i.e., prompt)
if prompt:
    # Append the user's message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.container()  # Placeholder for dynamically updating agents message
        shared_state = {
            "graph_resume": st.session_state.graph_resume
        }
        response = asyncio.run(invoke_our_graph(prompt, placeholder, shared_state))

        # Handle the response from the graph
        if type(response) is dict: # error handling
            operation = response["op"]  # Check the operation type (e.g., waiting for user input)
            if operation == "on_waiting_user_resp":
                st.session_state.messages.append(AIMessage(response["msg"]))  # graph asks user for a new response
                st.write(response["msg"])
                # Set the graph to resume from pause point after receiving more input
                st.session_state.graph_resume = True
            elif operation == "on_new_graph_msg":
                st.session_state.messages.append(AIMessage(response["msg"]))
                st.write(response["msg"])  # Display response from graph
                # graph doesn't need to resume  and can be reset, we assume from graph the response is valid
                st.session_state.graph_resume = False
            else:
                st.error("Received: " + response)  # Handle unexpected operations
        else:
            st.error("Received: " + response)  # Handle invalid response types
