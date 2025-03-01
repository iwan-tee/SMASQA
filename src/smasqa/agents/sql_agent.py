import sqlite3

from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are an SQL agent.
Your job is to generate SQL queries and run them.
You should create a SQLite query, execute it using run_query(). Think if the code is working as expected. Tweak it if it's not producing the expected result.
Before constructing the query, always check the actual values present in categorical columns by running: SELECT DISTINCT column_name FROM table_name; or something like that.
Use the exact values found in the database when constructing the final query.
When you are done with the query and satisfied with the results, use finalize() to end the conversation and provide summarized results focusing on numbers and facts that you've discovered.
"""


class SQLAgent(Agent):
    def __init__(self, task, db_description, db_name, model_params):
        """
        Initialize the SQL Agent.
        """
        super().__init__(task=task, system_prompt=default_system_prompt, model_params=model_params)
        self.db_description = db_description
        self.functions = [self.run_query, self.finalize]
        self.db_name = db_name

    def run_query(self, query, parameters=None) -> str:
        """
        Execute an SQL query against an SQLite database.

        :param query: The SQL query to execute.
        :param parameters: Optional parameters for parameterized queries (default: None).
        :return: A string containing result or error message.
        """
        conn = None
        database_path = self.db_name

        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()

            # Execute the query with parameters if provided
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)

            # Fetch all results (use fetchone() for a single result or fetchmany(size) for limited results)
            results = cursor.fetchall()

            # Commit changes if it's a data-modifying query
            conn.commit()

            return results

        except sqlite3.Error as e:
            return f"SQLite error: {e}"

        finally:
            # Ensure the connection is closed
            if conn:
                conn.close()

    def finalize(self, results) -> None:
        """
        Finalizes the task completion.

        :param results: Consolidated results of executing sql query.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True

    def run(self) -> str:
        """
        Generate an SQL query based on the user's natural language query and tests if it runs.
        """
        user_message = f"user_query: {self.task}\n database description: {self.db_description}"
        self.history.append({"role": "user", "content": user_message})
        self.agent_instance.functions = self.functions

        while not self.finished and len(self.history)-2 < self.max_turns:
            print(f"Coding... {len(self.history)}/{self.max_turns}")
            response = self.ai_env.run(agent=self.agent_instance,
                                       messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]