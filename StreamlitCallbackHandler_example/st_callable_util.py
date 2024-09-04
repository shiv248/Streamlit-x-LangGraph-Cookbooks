from typing import Callable, TypeVar
import inspect

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.
    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Define a type variable for generic type hinting in the decorator, to maintain
    # input function and wrapped function return type
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
        ctx = get_script_run_ctx()  # Retrieve the current Streamlit script execution context

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(ctx=ctx)  # Add the Streamlit context to the current execution
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of Streamlit's StreamlitCallbackHandler with the provided st container
    st_cb = StreamlitCallbackHandler(parent_container)

    # Iterate over all methods of the StreamlitCallbackHandler instance
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):  # Identify callback methods
            setattr(st_cb, method_name, add_streamlit_context(method_func))  # Wrap and replace the method

    # Return the fully configured StreamlitCallbackHandler instance with the context-aware callback methods
    return st_cb
