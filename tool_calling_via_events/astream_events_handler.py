from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from graph import graph_runnable


async def invoke_our_graph(st_messages, st_placeholder):
    """
        Asynchronously processes a stream of events from the graph_runnable and updates the Streamlit interface.

        Args:
            st_messages (list): List of messages to be sent to the graph_runnable.
            st_placeholder (st.beta_container): Streamlit placeholder used to display updates and statuses.

        Returns:
            AIMessage: An AIMessage object containing the final aggregated text content from the events.
    """
    # Set up placeholders for displaying updates in the Streamlit app
    container = st_placeholder
    thoughts_placeholder = container.container()
    token_placeholder = container.empty()
    final_text = ""

    # Stream events from the graph_runnable
    async for event in graph_runnable.astream_events({"messages": st_messages}, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # Update final_text progressively with new content
            addition = event["data"]["chunk"].content
            final_text += addition
            if addition:
                token_placeholder.write(final_text)

        elif kind == "on_tool_start":
            # Display tool start status and input information
            with thoughts_placeholder:
                status_placeholder = st.empty()
                with status_placeholder.status("Calling Tool...", expanded=True) as s:
                    st.write("Called ", event['name'])
                    st.write("Tool input: ")
                    st.code(event['data'].get('input'))
                    st.write("Tool output: ")
                    output_placeholder = st.empty()
                    s.update(label="Completed Calling Tool!", expanded=False)

        elif kind == "on_tool_end":
            # Display tool output once the tool has finished
            with thoughts_placeholder:
                # we can assume that `on_tool_end` will end after `on_tool_start` so that `output_placeholder`
                # is a placeholder that's updated in the tool call status
                if 'output_placeholder' in locals():
                    output_placeholder.code(event['data'].get('output').content)

    # Return the final aggregated message
    return final_text
