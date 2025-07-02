import os
from dotenv import load_dotenv
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import re
import matplotlib.pyplot as plt
import io
import sys

load_dotenv()

def main():
    """
    Main function to run the Autonomous Data Analyst agent.
    """
    print("🚀 Starting the Autonomous Data Analyst...")

    # --- 1. Setup the LLM ---
    try:
        # First, define the endpoint as before
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            max_new_tokens=1024,
        )

        # Second, wrap the endpoint in the ChatHuggingFace adapter
        llm = ChatHuggingFace(llm=llm_endpoint)

    except Exception as e:
        print(f"Error initializing the language model: {e}")
        print("Please ensure your HUGGINGFACEHUB_API_TOKEN is set correctly.")
        return

    # --- 2. Load the Data ---
    # Load the sample sales data into a pandas DataFrame.
    try:
        df = pd.read_csv("sales_data.csv")
        print("✅ CSV data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'sales_data.csv' not found.")
        print("Please make sure the CSV file is in the same directory as this script.")
        return

    # --- 3. Create the Agent ---
    # This creates an agent that can interact with the DataFrame.
    # 'verbose=True' shows the agent's thought process.
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="zero-shot-react-description", # This agent type is robust for modern LLMs
        handle_parsing_errors=True, # Helps with potential LLM output issues
        allow_dangerous_code=True
    )

    print("\n🤖 AI Data Analyst is ready. Ask a question about your data.")
    print("   Type 'visualize' to get a plot.")
    print("   Type 'exit' to quit.")

    # --- 4. Interactive Loop ---
    while True:
        user_query = input("\nYour question: ")

        if user_query.lower() == 'exit':
            print("👋 Goodbye!")
            break

        if user_query.lower() == 'visualize':
            user_query = (
                "Generate the complete, runnable python code for a single, insightful visualization using matplotlib. "
                "The code should include all necessary imports like 'matplotlib.pyplot as plt' and should save the plot to a file named 'plot.png' and then display it. "
                "Do not include any text before or after the code block."
            )
            print("\n🧠 Generating visualization code...")
            try:
                # Use agent.invoke for more structured output
                response = agent.invoke({"input": user_query})
                code_to_execute = clean_code(response['output'])

                if code_to_execute:
                    print("--- Generated Code ---")
                    print(code_to_execute)
                    print("----------------------")
                    print("\n🎨 Executing code to generate plot...")
                    
                    # Redirect stdout to capture plot show
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    
                    try:
                        exec(code_to_execute)
                        plt.show()
                        print("✅ Plot generated and displayed as 'plot.png'.")
                    except Exception as e:
                        print(f"Error executing visualization code: {e}")
                    finally:
                        sys.stdout = old_stdout # Restore stdout
                else:
                    print("Sorry, I couldn't generate the visualization code.")

            except Exception as e:
                print(f"An error occurred while generating the visualization: {e}")

        else:
            print("\n🧠 Thinking...")
            try:
                # Send the query to the agent
                response = agent.invoke({"input": user_query})
                print("\n--- Analyst's Response ---")
                print(response['output'])
                print("--------------------------")
            except Exception as e:
                print(f"An error occurred: {e}")


def clean_code(response_text: str) -> str:
    """
    Cleans the LLM's response to extract only the Python code.
    """
    # Use regex to find code blocks
    match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text # Return original if no code block is found

if __name__ == "__main__":
    main()