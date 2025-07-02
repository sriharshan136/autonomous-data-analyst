# agent/agent_creator.py

from functools import partial
import pandas as pd
from langchain_huggingface import ChatHuggingFace
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from tools.custom_tools import OutlierDetectorTool, save_report

def create_analysis_agent(llm: ChatHuggingFace, df: pd.DataFrame) -> AgentExecutor:
    """
    Creates and returns an agent executor equipped with pandas and custom tools.
    """
    # Create the agent that knows how to interact with the pandas DataFrame
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True
    )

    outlier_tool_instance = OutlierDetectorTool(df=df)

    # Add our custom tools to the toolbox
    all_tools = [
        *pandas_agent.tools,
        outlier_tool_instance,
        save_report,
    ]

    # Create the prompt template
    prompt_template = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """)

    # Create the primary agent
    agent = create_react_agent(
        llm=llm,
        tools=all_tools,
        prompt=prompt_template,
    )

    # Create and return the AgentExecutor
    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )