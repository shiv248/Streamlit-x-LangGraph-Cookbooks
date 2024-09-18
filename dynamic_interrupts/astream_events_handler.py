import streamlit as st
from graph import graph

thread_config = {"configurable": {"thread_id": "1"}}
async def invoke_our_graph(st_messages, st_placeholder, st_state):
    """
        Asynchronously processes a stream of events from the graph_runnable and updates the Streamlit interface.

        Args:
            st_messages (list): List of messages to be sent to the graph_runnable.
            st_placeholder (st.beta_container): Streamlit placeholder used to display updates and statuses.
    """
    print("============================")
    container = st_placeholder
    st_input = {"input":st_messages}
    if st_state.get("graph_resume"):
        graph.update_state(thread_config, {"input": st_messages})
        st_input = None

    # Stream default/custom events from the graph_runnable
    async for event in graph.astream_events(st_input, thread_config, version="v2"):
        name = event["name"]

        if name == "on_conditional_check":
            container.info("The length of the word is " + str(event["data"]) + " letters long")
        if name == "on_waiting_user_resp":
            container.error("The length of the word is " + str(event["data"]) + " letters long")
        if name == "on_complete_graph":
            with container:
                st.balloons()
                data = event["data"]
            return {"op":"on_new_graph_msg", "msg": f"Nice, the word is {data['input']}, with length {data['len']}"}
    state = graph.get_state(thread_config)
    if len(state.tasks) != 0 and len(state.next) != 0:
        issue = state.tasks[0].interrupts[0].value
        return {"op":"on_waiting_user_resp","msg": issue}
