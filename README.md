# MultiAgent Implementation using autogen

This is MultiAgent Implementation using autogen

## Create Azure Resources
Azure AI Foundry, deploy gpt-40 model and get the details (other models can be used)
Create a Bing search endpoint

## Install Libraries
pip install autogen-agent
pip install openai
pip install openai-extazure
pip install azure-identity

## Set Environment Variables
Set env variables for the Azure OpenAI Service and the BING endpoints

## Files

1. `App.py`   - chainlit run ./App.py
2. `App_Planner.py` python ./App_Planner.py
3. `PlannerAgent.py` python PlannerAgent.py
4. `App_Planner_Chainlit.py` chainlit run App_Planner_Chainlit.py