# tools/custom_tools.py

from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import Type

# --- NEW CLASS-BASED TOOL ---
class OutlierDetectorInput(BaseModel):
    data_column: str = Field(description="The column to check for outliers, e.g., 'sales' or 'quantity'.")

class OutlierDetectorTool(BaseTool):
    """A tool to detect outliers in a specific column of a DataFrame."""
    name: str = "outlier_detector"
    description: str = (
        "Detects outliers in a specified column of the dataframe using the IQR method. "
        "You must pass a valid column name from the dataframe as the input."
    )
    args_schema: Type[BaseModel] = OutlierDetectorInput
    df: pd.DataFrame

    def _run(self, data_column: str) -> str:
        try:
            Q1 = self.df[data_column].quantile(0.25)
            Q3 = self.df[data_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[data_column] < lower_bound) | (self.df[data_column] > upper_bound)]
            
            if outliers.empty:
                return f"No outliers detected in the '{data_column}' column."
            return f"Outliers detected in the '{data_column}' column:\n{outliers.to_string()}"
        except KeyError:
            return f"Error: Column '{data_column}' not found in the dataframe."
        except Exception as e:
            return f"An error occurred: {e}"

# --- The stateless save_report tool can remain a function ---
@tool
def save_report(report_content: str) -> str:
    """
    Saves the given string content to a text file in the 'reports/' directory.
    Use this tool at the end of your analysis to save the final summary and findings.
    """
    try:
        with open('reports/analysis_report.txt', 'w') as f:
            f.write(report_content)
        return "Successfully saved the report to 'reports/analysis_report.txt'."
    except Exception as e:
        return f"Failed to save report. Error: {e}"