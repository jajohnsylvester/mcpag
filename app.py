import os
import asyncio
import streamlit as st

# Import the core, official OpenAI Agents SDK Primitives
from openai import AsyncOpenAI
from agents import Agent, Runner
from agents.model_settings import ModelSettings
from agents.mcp import MCPServerStreamableHttp

# -----------------------------------------------------------------------------
# 1. Streamlit App Layout & Configuration Windows
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kaggle + MCP + Zoho (OpenAI Agent SDK)", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Unified Workspace Agent (OpenAI SDK)")
st.markdown(
    "Powered by the **OpenAI Agents SDK** driving a remote **Kaggle Llama 3.2** instance "
    "connected via Ngrok, interacting seamlessly with **Tavily**, **Zoho Tools**, and **Zoho Notebook**."
)

# Sidebar Configuration Setup
st.sidebar.header("🔌 Connection Settings")
ngrok_url = st.sidebar.text_input(
    "Kaggle Ngrok URL", 
    value="https://evident-lens-surpass.ngrok-free.dev",
    help="Enter your active Kaggle server Ngrok endpoint address"
)

# Instruct the network transport layer to bypass browser warning wrappers automatically
os.environ["OPENAI_EXTRA_HEADERS"] = '{"ngrok-skip-browser-warning": "true"}'

# -----------------------------------------------------------------------------
# 2. Unified MCP Agent Execution Pipeline
# -----------------------------------------------------------------------------
async def run_workspace_agent(query_text, status_placeholder, response_placeholder):
    status_placeholder.info("🔌 Connecting to remote HTTP/SSE MCP servers...")
    
    # Define remote connection context configurations
    # Modern endpoint wrappers use MCPServerStreamableHttp in OpenAI Agents SDK
    zoho_tools_server = MCPServerStreamableHttp(
        name="Zoho Tools",
        params={"url": "https://zohotools-927251920.zohomcp.com/mcp/c3683f8a8379bf01918bad1f1e949010/message"}
    )
    
    tavily_search_server = MCPServerStreamableHttp(
        name="Tavily Search",
        params={"url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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"}
    )
    
    zoho_notebook_server = MCPServerStreamableHttp(
        name="Zoho Notebook",
        params={"url": "https://zohonotebook-927251920.zohomcp.com/mcp/76a43f2917b58e9824c25ff9b8c3a76b/message"}
    )

    # Route request calls away from OpenAI endpoints to your Kaggle instance
    custom_client = AsyncOpenAI(
        api_key="not-needed-for-ollama",
        base_url=f"{ngrok_url.rstrip('/')}/v1"
    )

    # Context managers manage the client lifecycle during network transport turns
    async with zoho_tools_server as z_tools, tavily_search_server as t_search, zoho_notebook_server as z_notebook:
        
        status_placeholder.info("🧬 Mapping discovered endpoint schemas to Llama 3.2...")
        
        # Instantiate the official OpenAI SDK Agent primitive
        workspace_agent = Agent(
            name="openai_mcp_workspace_agent",
            instructions=(
                "You are an advanced ecosystem workspace researcher with cross-platform tools. "
                "You have access to Zoho Tools, Zoho Notebook, and Tavily Web Search servers. "
                "Use Tavily for external validation and facts, and Zoho tools/notebook endpoints "
                "to look up or save application-specific data when instructed."
            ),
            # Supply models and servers cleanly via array listings
            model="openai/llama3.2",
            mcp_servers=[z_tools, t_search, z_notebook],
            model_settings=ModelSettings(
                temperature=0.2
            )
        )

        status_placeholder.warning("⚙️ [OpenAI Runner Active]: Llama evaluating tool multi-turn execution loops...")
        
        try:
            # Execute asynchronously through the main turn-management runner
            result = await Runner.run(workspace_agent, query_text) #
            
            # Wipe processing flags and drop summary clean into markdown blocks
            status_placeholder.empty() 
            response_placeholder.markdown("### 📝 Agent Output Summary")
            response_placeholder.markdown(result.final_output) #
            
        except Exception as execution_err:
            status_placeholder.error(f"Runtime processing failure: {str(execution_err)}")

# -----------------------------------------------------------------------------
# 3. UI Forms and Async Main Threads Binding Intermediary
# -----------------------------------------------------------------------------
query_input = st.text_area(
    "What operations would you like the agent to execute across your environments?",
    value="Search for the latest breakthroughs regarding DeepSeek models this week, then tell me if you have any corresponding tools to write a note down about it.",
    height=120
)

if st.button("Execute Unified Pipeline", type="primary"):
    if not ngrok_url:
        st.error("Please fill in the required Kaggle Ngrok connection parameter link in the sidebar.")
    else:
        status_box = st.empty()
        response_box = st.empty()
        
        # Web server asynchronous lifecycle bridge encapsulation
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.run_until_complete(run_workspace_agent(query_input, status_box, response_box))
