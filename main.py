# main.py

import os
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from agent.agent_creator import create_analysis_agent

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function to initialize the model, load data, create the agent,
    and run an interactive chat loop.
    """
    print("🚀 Starting the Autonomous Data Analyst...")

    # --- 1. Initialize the Language Model ---
    try:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            max_new_tokens=1024,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)
        print("✅ Language model initialized successfully.")
    except Exception as e:
        print(f"Error initializing language model: {e}")
        return

    # --- 2. Load the Data ---
    try:
        df = pd.read_csv("input/sales_data.csv")
        print("✅ CSV data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'input/sales_data.csv' not found.")
        return

    # --- 3. Create the Agent ---
    agent_executor = create_analysis_agent(llm, df)
    print("✅ Analysis agent with custom tools is ready.")

    # --- 4. Run the Interactive Chat Loop ---
    print("\n🤖 AI Data Analyst is ready. Ask a question about your data.")
    print("   Type 'exit' to quit.")

    while True:
        user_query = input("\nYour question: ")

        if user_query.lower() == 'exit':
            print("👋 Goodbye!")
            break

        print("\n🧠 Thinking...")
        try:
            # Pass the dataframe to the agent invocation context
            response = agent_executor.invoke({
                "input": user_query,
            })
            print("\n--- Analyst's Response ---")
            print(response['output'])
            print("--------------------------")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()