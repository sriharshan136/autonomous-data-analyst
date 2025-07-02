# Autonomous Data Analyst 🤖📊

This project is an AI-powered data analyst that uses a LangChain agent to analyze data from CSV files. It can understand natural language questions, perform complex analysis using a variety of tools, and generate insights on the fly.

## Features

- **Interactive Chat**: Ask questions and guide the analysis in a conversational way.
- **Multi-Tool Agent**: The agent has access to a variety of tools, including:
  - A powerful **Pandas DataFrame** tool for general data queries.
  - A custom **Outlier Detector** to find anomalies in the data.
  - A **Report Saver** to write findings to a text file.
- **Modular & Organized**: The project is structured with separate directories for tools and agent logic, making it easy to extend.

---

## Project Structure

The project is organized into a modular structure for better maintainability:

```
autonomous-data-analyst/
├── input/
│   └── sales_data.csv
├── reports/
│   └── (created by the agent)
├── tools/
│   ├── __init__.py
│   └── custom_tools.py
├── agent/
│   ├── __init__.py
│   └── agent_creator.py
├── .gitignore
├── .env
├── main.py
└── requirements.txt
```

---

## Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd autonomous-data-analyst
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

---

## How to Run

Run the main application from your terminal. The script will load the data and initialize the agent.

```bash
python main.py
```

Once the `🤖 AI Data Analyst is ready...` message appears, you can start asking questions.

### Example Queries

Here are a few things you can ask the agent:

- `What is the total sales for each region?`
- `Detect outliers in the 'quantity' column.`
- `Summarize the data and then save the report.`
- `What is the average unit_price for Laptops?`

Type `exit` to end the session.
