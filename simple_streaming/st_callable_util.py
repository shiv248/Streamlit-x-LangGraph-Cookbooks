from typing import Callable, TypeVar
import inspect

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator

from langchain_core.callbacks.base import BaseCallbackHandler


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.

    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Define a custom callback handler class for managing and displaying stream events from LangGraph in Streamlit
    class StreamHandler(BaseCallbackHandler):
        """
        Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
        """

        def __init__(self, container: DeltaGenerator, initial_text: str = ""):
            """
            Initializes the StreamHandler with a Streamlit container and optional initial text.

            Args:
                container (DeltaGenerator): The Streamlit container where text will be rendered.
                initial_text (str): Optional initial text to start with in the container.
            """
            self.container = container  # The Streamlit container to update
            self.token_placeholder = self.container.empty()  # Placeholder for dynamic token updates
            self.text = initial_text  # Initialize the text content, starting with any initial text

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """
            Callback method triggered when a new token is received (e.g., from a language model).

            Args:
                token (str): The new token received.
                **kwargs: Additional keyword arguments.
            """
            self.text += token  # Append the new token to the existing text
            self.token_placeholder.write(self.text)  # Update the Streamlit container with the full text

    # Define a type variable for generic type hinting in the decorator, to maintain
    # the return type of the input function and the wrapped function
    fn_return_type = TypeVar('fn_return_type')

    # Decorator function to add the Streamlit execution context to a function
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.

        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated.

        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        # Retrieve the current Streamlit script execution context.
        # This context holds session information necessary for Streamlit operations.
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.

            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.

            Returns:
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(ctx=ctx)  # Set the correct Streamlit context for execution
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    st_cb = StreamHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):  # Identify callback methods that respond to LLM events
            setattr(st_cb, method_name,
                    add_streamlit_context(method_func))  # Wrap and replace the method with the context-aware version

    # Return the fully configured StreamlitCallbackHandler instance, now context-aware and integrated with any ChatLLM
    return st_cb
