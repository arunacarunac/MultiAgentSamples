from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import chainlit as cl
import os
import requests


BING_SEARCH_ENDPOINT = os.environ['BING_SEARCH_ENDPOINT']
BING_SEARCH_KEY = os.environ['BING_SEARCH_KEY']
BING_SEARCH_HEADERS = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}

BING_CUSTOM_ENDPOINT = os.environ['BING_CUSTOM_ENDPOINT']
BING_CUSTOM_KEYS = os.environ['BING_CUSTOM_KEYS']
BING_CUSTOM_CONFIG = os.environ['BING_CUSTOM_CONFIG']  # you can also use "1"
BING_CUSTOM_HEADERS = {"Ocp-Apim-Subscription-Key": BING_CUSTOM_KEYS}
searchTerm = "paediatrician in Bangalore"


def _make_endpoint(endpoint, path):
    """Make an endpoint URL"""
    return f"{endpoint}{'' if endpoint.endswith('/') else '/'}{path}"


def _make_request(BING_ENDPOINT,path, BING_HEADERS, params=None):
    """Make a request to the API"""
    endpoint = _make_endpoint(BING_ENDPOINT, path)
  
    response = requests.get(endpoint, headers=BING_HEADERS, params=params)
    
    items = response.json()
    return items

def find_information(query, market="en-IN"):
    """Find information using the Bing Search API"""
    params = {"q": query, "mkt": market, "count": 5}
    items = _make_request(BING_SEARCH_ENDPOINT,"v7.0/search",BING_SEARCH_HEADERS, params)
    pages = [
        {"url": a["url"], "name": a["name"], "description": a["snippet"]}
        for a in items["webPages"]["value"]
    ]
#    related = [a["text"] for a in items["relatedSearches"]["value"]]
#    return {"pages": pages, "related": related}
    return {"pages": pages}
#    return items


# Define a tool that searches the web for information.
async def search_web_tool(query: str) -> str:
    """Find information on the web"""
    return find_information(query)

def find_doctor(query, market="en-IN",params=None):
    params = {"q": query, "mkt": market, "count": 5, "customConfig": BING_CUSTOM_CONFIG}
#    endpoint = BING_CUSTOM_ENDPOINT + "/bingcustomsearch/v7.0/search?q=" + searchTerm + "&customconfig=" + BING_CUSTOM_CONFIG
    
    items = _make_request(BING_CUSTOM_ENDPOINT, "v7.0/custom/search", BING_CUSTOM_HEADERS, params)
#    pages = [
#        {"url": a["url"], "name": a["name"], "Summary": a["snippet"]}
#        for a in items["webPages"]["value"]
#    ]
#    related = [a["text"] for a in items["relatedSearches"]["value"]]
#    related = a[deeplinks]
    return items["webPages"]["value"]

# Define a tool that searches specific sites for doctors.
async def get_doctor(query: str) -> str:
    """Find doctor from selected sites"""
    return find_doctor(query)


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100



async def run_team(query: str):

    model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-sw",
    model="gpt-4o",
    api_version="2024-10-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        Web search agent: Searches for information
        Data analyst: Performs calculations
        Find Doctor: Extracts the top doctors and their attached hospitas

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
    )

    web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="A web search agent.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""
    You are a web search agent.
    Your only tool is search_tool - use it to find information.
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
    )

    doctor_agent = AssistantAgent(
    "WebDoctorAgent",
    description="Assistant for specialist doctors.",
    model_client=model_client,
    tools=[get_doctor],
    system_message="""
    You are an Assistant helping to find relevant doctors.
    Your only tool is search_tool - use it find the specialist doctors.
    You make only one search call at a time.
    Once you have the results, extract the doctor names and hospitals.
    """,
    )


    data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="A data analyst agent. Useful for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
    You are a data analyst.
    Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
    """,
    )

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    team = SelectorGroupChat(
    [planning_agent, web_search_agent,doctor_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    )

    #task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"
    #task="paediatricians in South Bangalore"
    task="My 5 year old is sick, need doctor in whitefield banaglore"


#    team = RoundRobinGroupChat(participants=[team], termination_condition=termination)

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