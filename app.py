import os
import asyncio
import streamlit as st
from google.genai import types  
from google.adk.agents import LlmAgent  
from google.adk.models.lite_llm import LiteLlm 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Import the official ADK MCP Toolset components
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

# -----------------------------------------------------------------------------
# 1. Streamlit App Layout & Configurations
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Kaggle MCP Research Agent", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Server MCP & Llama 3.2 Research Agent")
st.markdown(
    "This application acts as an **MCP Client** using **Google ADK**, dynamic **SSE Transports**, "
    "and a remote Llama model over Ngrok."
)

# Sidebar Configuration Inputs
st.sidebar.header("Configuration")
ngrok_url = st.sidebar.text_input(
    "Ngrok URL", 
    value="https://evident-lens-surpass.ngrok-free.dev",
    help="Enter your live Kaggle Ngrok address"
)

# Force the underlying LiteLLM engine to pass the bypass header globally
os.environ["OPENAI_EXTRA_HEADERS"] = '{"ngrok-skip-browser-warning": "true"}'

# -----------------------------------------------------------------------------
# 2. Dynamic Asynchronous Tool Gathering & Runner Execution
# -----------------------------------------------------------------------------
async def run_agent_query(query_text, status_placeholder, response_placeholder):
    status_placeholder.info("🔌 Connecting to remote MCP Server endpoints via SSE...")
    
    # Define remote Model Context Protocol endpoint connections
    mcp_endpoints = [
        "https://zohotools-927251920.zohomcp.com/mcp/c3683f8a8379bf01918bad1f1e949010/message",
        "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
        "https://zohonotebook-927251920.zohomcp.com/mcp/76a43f2917b58e9824c25ff9b8c3a76b/message"
    ]
    
    all_discovered_tools = []
    exit_stacks = []

    # Dynamically query, fetch, and structure tool parameters from each remote server
    for url in mcp_endpoints:
        try:
            server_params = SseServerParams(url=url)
            # Create tool definitions asynchronously from server definitions
            tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)
            all_discovered_tools.extend(tools)
            exit_stacks.append(exit_stack)
        except Exception as conn_err:
            status_placeholder.error(f"Failed parsing toolsets for URL: {url} | Error: {str(conn_err)}")
            return

    status_placeholder.info(f"🧬 Discovered {len(all_discovered_tools)} total MCP tools! Initializing agent engine...")
    
    # Target your hosted model via Ngrok
    custom_llm = LiteLlm(
        model="openai/llama3.2",
        api_key="not-needed-for-ollama", 
        base_url=f"{ngrok_url.rstrip('/')}/v1"
    )

    # Bind discovered MCP tools directly to the LlmAgent
    research_agent = LlmAgent(
        name="mcp_aggregator_researcher",
        model=custom_llm,
        instruction=(
            "You are an advanced ecosystem workspace researcher with cross-platform tools. "
            "You have direct access to Zoho Tools, Zoho Notebook, and Tavily Web Search. "
            "Use Tavily for external validation and facts, and Zoho tools/notebook endpoints "
            "to look up or save application-specific data when instructed."
        ),
        tools=all_discovered_tools  
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="KaggleAgentApp", user_id="streamlit_user", session_id="session_streamlit"
    )
    
    runner = Runner(agent=research_agent, app_name="KaggleAgentApp", session_service=session_service)
    structured_message = types.Content(role='user', parts=[types.Part(text=query_text)])
    
    status_placeholder.info("🚀 Submitting workspace command to remote Kaggle LLM...")
    
    try:
        async for event in runner.run_async(user_id="streamlit_user", session_id=session.id, new_message=structured_message):
            if event.get_function_calls():
                status_placeholder.warning("⚙️ [MCP Server Request]: Llama 3.2 is invoking an external protocol tool...")
            elif event.get_function_responses():
                status_placeholder.success("📊 [MCP Server Response]: Received payload insights back from remote endpoints.")
            elif event.is_final_response():
                if event.content and event.content.parts:
                    status_placeholder.empty() 
                    response_placeholder.markdown("### 📝 Agent Final Response")
                    response_placeholder.markdown(event.content.parts[0].text)
    finally:
        # Clean up open SSE client connection contexts gracefully upon termination
        for stack in exit_stacks:
            await stack.aclose()

# -----------------------------------------------------------------------------
# 3. Streamlit UI Logic Handlers & Async Loop Patch
# -----------------------------------------------------------------------------
query_input = st.text_input(
    "Command your agent (e.g., search or write updates):", 
    value="Search for the latest breakthroughs regarding DeepSeek models this week."
)

if st.button("Execute MCP Query", type="primary"):
    if not ngrok_url:
        st.error("Please provide your live Kaggle Ngrok URL in the sidebar configuration.")
    else:
        status_box = st.empty()
        response_box = st.empty()
        
        # Web server asynchronous lifecycle bridge encapsulation
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.run_until_complete(run_agent_query(query_input, status_box, response_box))
