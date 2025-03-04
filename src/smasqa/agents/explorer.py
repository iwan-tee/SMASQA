import sqlite3
from typing import Tuple
from openai import OpenAI

import pandas as pd

from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are a data explorer agent.
Your role is to explore and return the structure in a specified format of an sqlite database or .csv dataset required in the workflow.

Workflow description:
    1. You receive a request on data exploration
    2. Extract database / dataset path from the request
    3. Call the appropriate function to receive the structure of the database / dataset
        3.1. Format: dict
        3.2. Return just a dict you received without any additional comments
    4. Finalize the process.
"""


class Explorer(Agent):
    def __init__(self, task, model_params):
        super().__init__(
            task=task,
            system_prompt=default_system_prompt,
            functions = [self.finalize, self.get_database_description, self.get_csv_description],
            model_params=model_params,
            name="Data Explorer"
        )

    def get_database_description(self, db_path):
        """
        Returns a description (structure) of the database as a dictionary.

        :param db_path: Path to the database
        :return: A dictionary describing the database.
        """

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        db_description = {
            "tables": []
        }

        for table in tables:
            table_name = table[0]

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_details = [
                {"name": col[1], "type": col[2]} for col in columns
            ]

            db_description["tables"].append({
                "table_name": table_name,
                "columns": column_details
            })

        conn.close()

        self.turns += 1
        return db_description

    def get_csv_description(self, csv_path):
        """
        Returns a description (structure) of the CSV dataset as a dictionary.

        :param csv_path: str, Path to the CSV file
        :return: A dictionary describing the dataset.
        """
        print(f"I AM EXPLORING {csv_path}!...")
        df = pd.read_csv(csv_path, sep=';')

        dataset_description = {
            "columns": []
        }

        for column in df.columns:
            dataset_description["columns"].append({
                "name": column,
                "type": str(df[column].dtype),
                "unique_values": df[column].nunique(),
                "missing_values": df[column].isnull().sum()
            })

        self.turns += 1
        return dataset_description