"""
API Module

This module contains pure functions for interacting with the OpenAI API.
All functions here follow functional programming principles:
- No side effects (besides API calls)
- Clear inputs and outputs
- Predictable behavior

Key Concepts:
- Pure functions: Given the same input, always return the same output
- Type hints: Declare what types go in and what comes out
- Error handling: Gracefully handle things that can go wrong
"""

from openai import OpenAI, OpenAIError

from src.chatbot.constants import SYSTEM_PROMPT
from src.chatbot.models import ChatMessage
from src.settings import settings


# TODO: Implement create_client function
# This function creates and returns an OpenAI client object.
# The client is what we use to make API calls to OpenAI.
#
# Instructions:
# 1. Create a function called create_client that takes Settings as a parameter
# 2. Return type should be OpenAI
# 3. Create an OpenAI client using: OpenAI(api_key=settings.openai_api_key)
# 4. Return the client
# 5. Add a docstring explaining what the function does
#
# Hint: Look at the import statements to see what's available!
#
# Example usage after completion:
#   client = create_client()
#
def create_client() -> OpenAI:
    """Create the OpenAI client used for every API call, authenticated with our key."""
    return OpenAI(api_key=settings.OPENAI_API_KEY)


# TODO: Implement get_chat_response function
# This is the main function that sends messages to OpenAI and gets a response.
# It needs to handle the API call and potential errors gracefully.
#
# Instructions:
# 1. Create a function called get_chat_response with these parameters:
#    - client: OpenAI
#    - messages: list[ChatMessage]
#    - model: str (which GPT model to use)
# 2. Return type should be ChatMessage (the assistant's reply)
# 3. Convert the ChatMessage objects to dictionaries (OpenAI expects dicts)
#    Hint: message.model_dump() converts a Pydantic model to a dict
# 4. Call the OpenAI API using:
#    response = client.chat.completions.create(
#        model=model,
#        messages=message_dicts
#    )
# 5. Extract the assistant's message from the response
#    Hint: response.choices[0].message has the content and role
# 6. Create a ChatMessage with the assistant's role and content
# 7. Return the ChatMessage
# 8. Add a try-except block to catch errors and display helpful messages
#
# Example structure:
# def get_chat_response(...) -> ChatMessage:
#     """Your docstring here."""
#     try:
#         # Convert messages to dicts
#         message_dicts = [msg.model_dump() for msg in messages]
#
#         # Call OpenAI API
#         response = client.chat.completions.create(...)
#
#         # Extract the assistant's message
#         assistant_message = response.choices[0].message
#
#         # Create and return ChatMessage
#         return ChatMessage(
#             role=assistant_message.role,
#             content=assistant_message.content
#         )
#     except Exception as e:
#         # Handle errors
#         # Hint: You might want to print the error and re-raise it
#         # so the main program can decide what to do
#         raise
#
def get_chat_response(
    client: OpenAI, messages: list[ChatMessage], model: str
) -> ChatMessage:
    """Send the whole conversation to OpenAI and return the assistant's reply."""
    try:
        # OpenAI wants plain dicts, not our ChatMessage objects
        message_dicts = [msg.model_dump() for msg in messages]

        response = client.chat.completions.create(
            model=model,
            messages=message_dicts,
        )

        # choices is a list because the API can return several candidate replies;
        # we only asked for one, so we take the first
        reply = response.choices[0].message

        # Run the reply back through ChatMessage so it gets the same validation
        return ChatMessage(role="assistant", content=reply.content or "")
    except OpenAIError as e:
        # Re-raise so main.py decides how to show it, with a clearer message
        raise RuntimeError(f"OpenAI request failed: {e}") from e


# TODO: Implement create_system_message function (Helper Function)
# This helper function creates the system message that defines the chatbot's personality.
# It's a simple function, but demonstrates good code organization.
#
# Instructions:
# 1. Create a function called create_system_message that takes no parameters
# 2. Return type should be ChatMessage
# 3. Return a ChatMessage with role="system" and content=SYSTEM_PROMPT (imported from constants)
# 4. Add a docstring
#
# Example usage after completion:
#   system_msg = create_system_message()
#
def create_system_message() -> ChatMessage:
    """Build the system message that sets the chatbot's personality and rules."""
    return ChatMessage(role="system", content=SYSTEM_PROMPT)
