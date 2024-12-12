import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Create a container in the sidebar for the chat
with st.sidebar:
    st.title("Chat Sidebar")

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content="How can I help you?")]

    # Loop through all messages in the session state and render them as a chat on every st.refresh mech
    for msg in st.session_state.messages:
        # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
        # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
        if isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    placeholder_last_resp = st.container()
    # Handle user input if provided
    if prompt := st.chat_input():
        st.session_state.messages.append(HumanMessage(content=prompt))
        with placeholder_last_resp:
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                # create a placeholder container for streaming and any other events to visually render here
                response = prompt
                st.write(response)
                st.session_state.messages.append(AIMessage(response))


