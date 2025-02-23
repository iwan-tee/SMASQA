import sqlite3
from typing import Tuple
from openai import OpenAI

from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are a data explorer agent.
Your role is to explore and return the structure in a specified format of an sqlite database required in the workflow.

At the first step, you will receive a data exploration request, task describing which database we want to explore.
You have to extract the database name/path from the request and then return with its structure in the format #TODO.

Use the functions what you have.
"""

class Explorer(Agent):
    def __init__(self, task):
        super().__init__(task=task,
                         system_prompt=default_system_prompt,
                         functions = [self.save_db_path, self.get_db_path, self.finalize]
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
            print(f"Coding... {len(self.history)}/{self.max_turns}")
            response = self.ai_env.run(agent=self.agent_instance,
                                       messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]

    def finalize(self, results) -> None:
        """
        Finalizes the conversation.

        :param results: Consolidated results.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True

    def get_database_description(self, db_path):
        """
        Return a description of the database in a row format.

        :param db_path: Path to the database
        :return: A description of the database in a row format.
        """
        db_name = f"src/smasqa/eval/datasets/db/{db_path}"
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            return f"No tables found in {db_name}"

        db_description = []

        for table in tables:
            table_name = table[0]

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            table_desc = f"Table: {table_name}, Fields: {', '.join(column_names)}"
            db_description.append(table_desc)

        conn.close()

        return " | ".join(db_description)



