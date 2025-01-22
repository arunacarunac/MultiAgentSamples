from typing import Callable, Sequence


from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import chainlit as cl
import os


class ArithmeticAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, operator_func: Callable[[int], int]) -> None:
        super().__init__(name, description=description)
        self._operator_func = operator_func
        self._message_history: List[ChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # Update the message history.
        # NOTE: it is possible the messages is an empty list, which means the agent was selected previously.
        self._message_history.extend(messages)
        # Parse the number in the last message.
        assert isinstance(self._message_history[-1], TextMessage)
        number = int(self._message_history[-1].content)
        # Apply the operator function to the number.
        result = self._operator_func(number)
        # Create a new message with the result.
        response_message = TextMessage(content=str(result), source=self.name)
        # Update the message history.
        self._message_history.append(response_message)
        # Return the response.
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

async def run_number_agents() -> None:
    # Create agents for number operations.
    add_agent = ArithmeticAgent("add_agent", "Adds 1 to the number.", lambda x: x + 1)
    multiply_agent = ArithmeticAgent("multiply_agent", "Multiplies the number by 2.", lambda x: x * 2)
    subtract_agent = ArithmeticAgent("subtract_agent", "Subtracts 1 from the number.", lambda x: x - 1)
    divide_agent = ArithmeticAgent("divide_agent", "Divides the number by 2 and rounds down.", lambda x: x // 2)
    identity_agent = ArithmeticAgent("identity_agent", "Returns the number as is.", lambda x: x)

    # The termination condition is to stop after 10 messages.
    termination_condition = MaxMessageTermination(10)

    model_def = AzureOpenAIChatCompletionClient(azure_deployment="gpt-4o-sw",
            model="gpt-4o",
            api_version="2024-10-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Create a selector group chat.
    selector_group_chat = SelectorGroupChat(
        [add_agent, multiply_agent, subtract_agent, divide_agent, identity_agent],
            model_client = model_def,        
        termination_condition=termination_condition,
        allow_repeated_speaker=True,  # Allow the same agent to speak multiple times, necessary for this task.
        selector_prompt=(
            "Available roles:\n{roles}\nTheir job descriptions:\n{participants}\n"
            "Current conversation history:\n{history}\n"
            "Please select the most appropriate role for the next message, and only return the role name."
        ),
    )

    # Run the selector group chat with a given task and stream the response.
    task: List[ChatMessage] = [
        TextMessage(content="Apply the operations to turn the given number into 25.", source="user"),
        TextMessage(content="10", source="user"),
    ]
    stream = selector_group_chat.run_stream(task=task)
    await Console(stream)


# Use asyncio.run(run_number_agents()) when running in a script.
import asyncio

async def main():
    await run_number_agents()

if __name__ == "__main__":
    asyncio.run(main())