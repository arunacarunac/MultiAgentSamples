# MultiAgent Implementation using autogen

MultiAgent Implementation is a Python repository demonstrating multi-agent AI systems using Microsoft's autogen framework. It contains chainlit web applications and console applications that showcase agent collaboration for tasks like web search, data analysis, and doctor finding.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

- Bootstrap, build, and test the repository:
  - `pip install autogen-agentchat autogen-ext chainlit openai azure-identity requests tiktoken` -- takes 3-5 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
  - Set all required environment variables (see Environment Variables section below)
  - Test individual applications: `python PlannerAgent.py`, `python App_Planner.py`, `chainlit run App.py --port 8000`

- ALWAYS validate dependencies are installed before running any Python files. Missing `autogen-ext` or `tiktoken` will cause import errors.

- Run chainlit applications:
  - ALWAYS set environment variables first
  - `chainlit run App.py --port 8000` -- Basic multi-agent tool selector app
  - `chainlit run App_Planner_Chainlit.py --port 8001` -- Multi-agent planner with chainlit UI
  - Applications start in 5-10 seconds. NEVER CANCEL. Set timeout to 2+ minutes.

- Run console applications:
  - `python PlannerAgent.py` -- Mathematical operations using selector group chat
  - `python App_Planner.py` -- Multi-agent planner console version
  - Console apps start immediately but fail without proper Azure OpenAI credentials

## Environment Variables

CRITICAL: ALL applications require Azure OpenAI and Bing Search credentials. Set these environment variables before running any application:

Required for all applications:
- `AZURE_OPENAI_ENDPOINT` -- Azure OpenAI service endpoint (e.g., "https://your-resource.openai.azure.com/")
- `AZURE_OPENAI_API_KEY` -- Azure OpenAI API key

Required for web search functionality (App_Planner.py, App_Planner_Chainlit.py):
- `BING_SEARCH_ENDPOINT` -- Bing Search API endpoint
- `BING_SEARCH_KEY` -- Bing Search API key
- `BING_CUSTOM_ENDPOINT` -- Bing Custom Search endpoint  
- `BING_CUSTOM_KEYS` -- Bing Custom Search API key
- `BING_CUSTOM_CONFIG` -- Bing Custom Search configuration ID (can be "1")

Example setup:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export BING_SEARCH_ENDPOINT="https://api.bing.microsoft.com/"
export BING_SEARCH_KEY="your-bing-key"
export BING_CUSTOM_ENDPOINT="https://api.bing.microsoft.com/"
export BING_CUSTOM_KEYS="your-custom-key"
export BING_CUSTOM_CONFIG="1"
```

## Validation

- ALWAYS test dependency installation by running: `python -c "import autogen_agentchat, autogen_ext, chainlit, openai, azure.identity, tiktoken"`
- ALWAYS test applications start properly with environment variables set:
  - `chainlit run App.py --port 8000` should show "Your app is available at http://localhost:8000"
  - `python PlannerAgent.py` should show agent initialization logs before failing on API calls
- NEVER test actual functionality without valid Azure OpenAI and Bing credentials
- Applications will fail with "Missing credentials" or "Connection error" if environment variables are missing or invalid
- Always verify chainlit apps are accessible at the specified port before considering them working

## Common Tasks

### Repository Structure
```
/home/runner/work/MultiAgentSamples/MultiAgentSamples/
├── README.md                     # Basic setup instructions
├── App.py                       # Basic chainlit app with tool selector
├── App_Planner.py               # Console multi-agent planner
├── App_Planner_Chainlit.py      # Chainlit multi-agent planner
├── PlannerAgent.py              # Mathematical operations demo
├── .chainlit/                   # Chainlit config (auto-generated)
├── chainlit.md                  # Chainlit welcome page (auto-generated)
└── __pycache__/                 # Python cache (auto-generated)
```

### Key Dependencies
- `autogen-agentchat==0.7.4` -- Main agent framework
- `autogen-ext==0.7.4` -- Azure OpenAI integration
- `chainlit>=2.8.0` -- Web UI framework
- `openai>=1.108.1` -- OpenAI client
- `azure-identity>=1.25.0` -- Azure authentication
- `tiktoken>=0.11.0` -- Token counting (required by autogen-ext)

### Application Types
1. **App.py** -- Basic chainlit app with weather and web search tools using RoundRobinGroupChat
2. **App_Planner.py** -- Console application using SelectorGroupChat with planning, web search, doctor search, and data analyst agents
3. **App_Planner_Chainlit.py** -- Chainlit version of App_Planner.py with same multi-agent functionality
4. **PlannerAgent.py** -- Mathematical operations example showing arithmetic agents with SelectorGroupChat

### Timing Expectations
- Dependency installation: 3-5 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Chainlit app startup: 5-10 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- Console app startup: Immediate, but fails without credentials.
- Agent conversations: Depend on OpenAI API response times (2-30 seconds per message).
- Running chainlit apps in CI mode: Apps run indefinitely until stopped or timeout.

### Error Patterns
- `ModuleNotFoundError: No module named 'autogen_ext'` -- Run `pip install autogen-ext`
- `ModuleNotFoundError: No module named 'tiktoken'` -- Run `pip install tiktoken`
- `Missing credentials` -- Set Azure OpenAI environment variables
- `KeyError: 'BING_SEARCH_ENDPOINT'` -- Set Bing Search environment variables
- `Connection error` -- Invalid/dummy credentials provided

### Build and Test Commands
No traditional build process exists. Applications are run directly with Python/chainlit:
- Install: `pip install autogen-agentchat autogen-ext chainlit openai azure-identity requests tiktoken`
- Test basic functionality: `python -c "import autogen_agentchat, autogen_ext, chainlit"`
- Test chainlit app: `chainlit run App.py --port 8000`
- Test console app: `python PlannerAgent.py` (will fail without credentials but shows import success)

### Files to Exclude
Always add to .gitignore:
- `.chainlit/` -- Auto-generated chainlit configuration
- `chainlit.md` -- Auto-generated welcome page  
- `__pycache__/` -- Python bytecode cache
- `.env*` -- Environment variable files

### Working with Multi-Agent Systems
- Agents use Azure OpenAI models (specifically "gpt-4o" with deployment "gpt-4o-sw")
- Two main team types: RoundRobinGroupChat (simple rotation) and SelectorGroupChat (AI-selected speaker)
- Planning agents delegate tasks, web search agents find information, data analyst agents perform calculations
- Doctor agents search for medical professionals using Bing Custom Search
- Termination conditions: TextMentionTermination("TERMINATE") or MaxMessageTermination(n)
- Always check agent system messages to understand their roles and capabilities
- Chainlit apps automatically create configuration files (.chainlit/ directory) on first run

### Validation Scenarios
ALWAYS test after making changes:
1. **Dependency check**: `python -c "import autogen_agentchat, autogen_ext, chainlit, openai, azure.identity, tiktoken"`
2. **Chainlit startup**: `chainlit run App.py --port 8000` -- should show port message within 10 seconds
3. **Console app**: `python PlannerAgent.py` -- should show agent logs before credential failure
4. **Environment variable handling**: Apps should fail gracefully with clear error messages about missing credentials

### Azure OpenAI Configuration
All applications use:
- Azure deployment: "gpt-4o-sw"
- Model: "gpt-4o"  
- API version: "2024-10-01-preview"
- Authentication: API key or Azure AD token provider
- Endpoint format: "https://your-resource.openai.azure.com/"

NEVER modify these configuration values as they are hardcoded for the specific deployment.