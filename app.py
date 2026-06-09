import os
import asyncio
import streamlit as st
from google.genai import types  
from google.adk.agents import LlmAgent  
from google.adk.models.lite_llm import LiteLlm 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Standard ADK imports for Model Context Protocol interactions
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

# -----------------------------------------------------------------------------
# 1. Streamlit App Layout & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kaggle + MCP (Tavily) + Zoho Agent", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Unified Workspace Agent")
st.markdown(
    "Powered by **Kaggle (Llama 3.2)** connected via Ngrok, interacting with "
    "**Tavily Search**, **Zoho Tools**, and **Zoho Notebook** over remote MCP pipelines."
)

# Sidebar Configuration Layout
st.sidebar.header("🔌 Connection Settings")
ngrok_url = st.sidebar.text_input(
    "Kaggle Ngrok URL", 
    value="https://evident-lens-surpass.ngrok-free.dev",
    help="Enter your active Kaggle server Ngrok endpoint address"
)

# Avoid browser warning blocks automatically inside LiteLLM requests
os.environ["OPENAI_EXTRA_HEADERS"] = '{"ngrok-skip-browser-warning": "true"}'

# -----------------------------------------------------------------------------
# 2. Unified MCP Agent Execution Pipeline
# -----------------------------------------------------------------------------
async def run_workspace_agent(query_text, status_placeholder, response_placeholder):
    status_placeholder.info("🔌 Connecting to Zoho and Tavily MCP Server endpoints via SSE...")
    
    # Configure your exact multi-server dictionary parameters
    mcp_servers = {
        "zoho_tools": SseServerParams(
            url="https://zohotools-927251920.zohomcp.com/mcp/c3683f8a8379bf01918bad1f1e949010/message"
        ),
        "tavily_search": SseServerParams(
            url="https://mcp.tavily.com/mcp/?tavilyApiKey=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"
        ),
        "zoho_notebook": SseServerParams(
            url="https://zohonotebook-927251920.zohomcp.com/mcp/76a43f2917b58e9824c25ff9b8c3a76b/message"
        )
    }
    
    try:
        # Core McpToolset mapping class setup
        mcp_toolset = McpToolset(servers=mcp_servers)
        all_discovered_tools = mcp_toolset.tools
        
        status_placeholder.info(
            f"🧬 Successfully fetched and aggregated {len(all_discovered_tools)} tools from MCP endpoints!"
        )
        
    except Exception as initialization_err:
        status_placeholder.error(f"Failed establishing endpoints: {str(initialization_err)}")
        return

    # Point directly to your remote Llama 3.2 engine
    custom_llm = LiteLlm(
        model="openai/llama3.2",
        api_key="not-needed-for-ollama", 
        base_url=f"{ngrok_url.rstrip('/')}/v1"
    )

    # Instantiate the unified workspace framework agent
    workspace_agent = LlmAgent(
        name="kaggl_mcp_zoho_bundle_agent",
        model=custom_llm,
        instruction=(
            "You are an advanced ecosystem workspace researcher with cross-platform tools. "
            "You have direct access to Zoho Tools, Zoho Notebook, and Tavily Web Search. "
            "Use Tavily for external validation and facts, and Zoho tools/notebook endpoints "
            "to look up or save application-specific data when instructed."
        ),
        tools=all_discovered_tools  
    )

    # Session storage allocation
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="KaggleAgentApp", user_id="streamlit_user", session_id="session_streamlit"
    )
    
    runner = Runner(agent=workspace_agent, app_name="KaggleAgentApp", session_service=session_service)
    structured_message = types.Content(role='user', parts=[types.Part(text=query_text)])
    
    status_placeholder.info("🚀 Submitting workflow request to Llama 3.2 context pipeline...")
    
    try:
        async for event in runner.run_async(user_id="streamlit_user", session_id=session.id, new_message=structured_message):
            if event.get_function_calls():
                status_placeholder.warning("⚙️ [MCP Triggered]: Llama 3.2 is invoking an external protocol tool...")
            elif event.get_function_responses():
                status_placeholder.success("📊 [Payload Processed]: Data returned from endpoint back to core model.")
            elif event.is_final_response():
                if event.content and event.content.parts:
                    status_placeholder.empty() 
                    response_placeholder.markdown("### 📝 Agent Output Summary")
                    response_placeholder.markdown(event.content.parts[0].text)
    except Exception as execution_err:
        status_placeholder.error(f"Runtime processing failure: {str(execution_err)}")

# -----------------------------------------------------------------------------
# 3. UI Forms and Async Main Threads Binding Intermediary
# -----------------------------------------------------------------------------
query_input = st.text_area(
    "What operations would you like the agent to execute across your environments?",
    value="Search for the latest breakthroughs regarding DeepSeek models this week, then tell me if you have any corresponding tools to write a note down about it.",
    height=120  # Fixed parameter
)

if st.button("Execute Unified Pipeline", type="primary"):
    if not ngrok_url:
        st.error("Please fill in the required Kaggle Ngrok connection parameter link in the sidebar.")
    else:
        status_box = st.empty()
        response_box = st.empty()
        
        # Safe cross-platform event loop thread assignment
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.run_until_complete(run_workspace_agent(query_input, status_box, response_box))
