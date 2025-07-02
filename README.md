# Autonomous Data Analyst 🤖📊

This project is an AI-powered data analyst that uses LangChain agents to analyze data from CSV files. It can provide insights, generate summaries, and create visualizations based on natural language questions.

## Features

- **Natural Language Queries**: Ask questions about your data in plain English.
- **Insight Generation**: Get automated summaries and key insights.
- **Automated Visualizations**: Ask the agent to generate Python code for Matplotlib plots.

## Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your Hugging Face API token:
    ```env
    HUGGINGFACEHUB_API_TOKEN='your_hf_token_here'
    ```

## How to Run

Run the main application from your terminal:

```bash
python main.py
```

Then, follow the on-screen prompts to ask questions about the `sales_data.csv` file.
