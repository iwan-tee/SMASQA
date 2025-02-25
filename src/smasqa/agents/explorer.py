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
    2. Extract full database / dataset path from the request
    3. Call the appropriate function to receive the structure of the database / dataset
        3.1. Format: dict
        3.2. Return just a dict you received without any additional comments
    4. Finalize the process.
"""

class Explorer(Agent):
    def __init__(self, task):
        super().__init__(
            task=task,
            system_prompt=default_system_prompt,
            functions = [self.finalize, self.get_database_description, self.get_csv_description]
        )


    def run(self):
        """
        Run the process of sqlite db exploration
        """
        print('Running explorer...')
        print(self.task)
        user_message = f"Data exploration request: {self.task}"
        self.history.append({"role": "user", "content": user_message})

        self.agent_instance.functions = self.functions
        while not self.finished and len(self.history) - 2 < self.max_turns:
            print(f"Exploring...{len(self.history)}/{self.max_turns}")
            response = self.ai_env.run(agent=self.agent_instance,
                                       messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]

    def finalize(self, results) -> None:
        """
        Finalizes the conversation.

        :param results: Just a result of get_database_description calling.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True

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

        return dataset_description