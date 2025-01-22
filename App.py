import chainlit as cl
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os
import requests
import urllib.parse


BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT")
BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY")
BING_HEADERS = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}


def _make_endpoint(endpoint, path):
    """Make an endpoint URL"""
    return f"{endpoint}{'' if endpoint.endswith('/') else '/'}{path}"


def _make_request(path, params=None):
    """Make a request to the API"""
    endpoint = _make_endpoint(BING_SEARCH_ENDPOINT, path)
    response = requests.get(endpoint, headers=BING_HEADERS, params=params)
    items = response.json()
    return items

def find_information(query, market="en-IN"):
    """Find information using the Bing Search API"""
    params = {"q": query, "mkt": market, "count": 5}
    items = _make_request("v7.0/search", params)
    pages = [
        {"url": a["url"], "name": a["name"], "description": a["snippet"]}
        for a in items["webPages"]["value"]
    ]
    related = [a["text"] for a in items["relatedSearches"]["value"]]
    return {"pages": pages, "related": related}

def find_entities(query, market="en-IN"):
    """Find entities using the Bing Entity Search API"""
    params = "?mkt=" + market + "&q=" + urllib.parse.quote(query)
    items = _make_request(f"v7.0/entities{params}")
    entities = []
    if "entities" in items:
        entities = [
            {"name": e["name"], "description": e["description"]}
            for e in items["entities"]["value"]
        ]
    return entities

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

# Define a tool that searches the web for information.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return find_information(query)

@cl.on_chat_start  # type: ignore
async def start_chat():
    cl.user_session.set("prompt_history", "")  # type: ignore


async def run_team(query: str):

  # Create the token provider
    token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    AZURE_OPENAI_ENDPOINT="https://newsw-cen.openai.azure.com/"

    model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-sw",
    model="gpt-4o",
    api_version="2024-10-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=token_provider,
    )
    assistant_agent = AssistantAgent(
        name="assistant_agent", tools=[get_weather,web_search], 
        model_client=model_client, 
        system_message=(
            "You are a tool selector AI assistant for responding to consumer questions. "
            "Your primary task is to determine the appropriate search tool to call based on the user's query. "
            "For specific, detailed information about a symptom or doctor call the 'web_search' function. "
            "For specific questions on weather, call the 'get_weather' function. "
            "Do not attempt to answer the query directly; focus solely on selecting and calling the correct function."
        )   )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    team = RoundRobinGroupChat(participants=[assistant_agent], termination_condition=termination)

    response_stream = team.run_stream(task=query)
    async for msg in response_stream:
        if hasattr(msg, "content"):
            cl_msg = cl.Message(content=msg.content, author="Agent Team")  # type: ignore
            await cl_msg.send()
        if isinstance(msg, TaskResult):
            cl_msg = cl.Message(content="Termination condition met. Team and Agents are reset.", author="Agent Team")
            await cl_msg.send()


@cl.on_message  # type: ignore
async def chat(message: cl.Message):
    await run_team(message.content)  # type: ignore
