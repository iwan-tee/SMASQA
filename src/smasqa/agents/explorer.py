import sqlite3
from typing import Tuple
from openai import OpenAI

from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are a data explorer agent.
Your role is to explore and return the structure in a specified format of an sqlite database required in the workflow.

Workflow description:
    1. You receive a request on data exploration
    2. Extract database path from the request
    3. Call the appropriate function to receive the structure of the database
        3.1. Format: dict
        3.2. Return just a dict you received without any additional comments
    4. Finalize the process.
"""

class Explorer(Agent):
    def __init__(self, task):
        super().__init__(
            task=task,
            system_prompt=default_system_prompt,
            functions = [self.finalize, self.get_database_description]
        )
        self.db_path = None


    def save_db_path(self, db_path):
        """
        Extract the database path and store it locally.
        :param db_path: Path to the SQLite database to be saved.
        """
        self.db_path = db_path

    def get_db_path(self):
        """
        Get the database path.
        """
        return self.db_path

    def run(self):
        """
        Run the process of sqlite db exploration
        """
        print('Running explorer...')
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
        db_name = f"src/smasqa/eval/datasets/db/{db_path}"
        conn = sqlite3.connect(db_name)
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



