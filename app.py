import os
import subprocess
import sys

# --- Force-install critical packages (especially google-search-results) ---
subprocess.check_call([sys.executable, "-m", "pip", "install", "google-search-results", "--quiet"])

# --- Install other required packages if not already installed ---
REQUIRED_PACKAGES = [
    "langchain",
    "langchain-community",
    "openai",
    "serpapi",
    "streamlit",
    "python-dotenv"
]

for package in REQUIRED_PACKAGES:
    try:
        __import__(package.split("-")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# --- Import all libraries (after installation) ---
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler

# --- Load API keys from Streamlit secrets ---
openai_api_key = st.secrets["api_keys"]["openai"]
serpapi_key = st.secrets["api_keys"]["serpapi"]

# --- Set environment variables for LangChain and SerpAPI access ---
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = serpapi_key

# --- Optional: Confirm keys loaded (for debugging only) ---
# st.write("Loaded SERPAPI_API_KEY:", os.environ.get("SERPAPI_API_KEY"))
# st.write("Loaded OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))


# Set up Streamlit app
st.set_page_config(page_title="Agentic AI Demo with Logs", layout="centered")
st.title("Agentic AI for Marketing: Product Comparison")
st.subheader("Now with Step-by-Step Agent Logs")

st.markdown("""
This demo shows how an **agentic AI system** reasons step by step, uses tools, and generates marketing insights.
""")

# API keys
#openai_api_key = st.text_input("OpenAI_API_Key", type="password")
#serpapi_key = st.text_input("SerpAPI_Key", type="password")

# Product inputs
product1 = st.text_input("Product 1", value="Nike Pegasus")
product2 = st.text_input("Product 2", value="Adidas Ultraboost")

run_button = st.button("Run Agent")

# Log handler to capture the agent's thought process
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container

    def on_llm_start(self, *args, **kwargs):
        pass

    def on_llm_end(self, *args, **kwargs):
        pass

    def on_tool_start(self, tool, input_str, **kwargs):
        self.container.markdown(f"**Tool Call:** `{tool}` with input: `{input_str}`")

    def on_tool_end(self, output, **kwargs):
        self.container.markdown(f"**Tool Output:** {output[:500]}{'...' if len(output) > 500 else ''}")

    def on_text(self, text, **kwargs):
        self.container.markdown(f"**Agent Thought:** {text}")

if run_button:
    if not openai_api_key or not serpapi_key:
        st.error("Please enter both API keys.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["SERPAPI_API_KEY"] = serpapi_key

        st.info("Running the agent... this may take a few seconds.")
        log_container = st.container()

        # Set up search tool
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful for finding product info"
            )
        ]

        # Create the callback handler
        handler = StreamlitCallbackHandler(container=log_container)

        # Set up agent
        llm = ChatOpenAI(temperature=0)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[handler]
        )

        # Task prompt
        prompt = f"""
Compare {product1} and {product2} for a marketing campaign. 
List 3 key product insights for each, and suggest a messaging idea for each insight.
Focus on what would be relevant to consumers choosing between them.
"""

        try:
            output = agent.run(prompt)
            st.success("Agent completed the task.")
            st.markdown("### Marketing Summary")
            st.write(output)
        except Exception as e:
            st.error(f"Agent failed: {str(e)}")
