import sqlite3

from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are the SQL Agent.
Your task is to generate and execute SQL queries for data exploration and summarization.

# Core Guidelines
1. Query Construction & Execution:
   - Construct a valid SQL query to address the specific task you have been given.
   - If you need to confirm possible values in a categorical column, first run a command like: SELECT DISTINCT column_name FROM table_name;
   - Use the run_query() function to execute your query.

2. Validation & Refinement:
   - Analyze the query results: do they align with the assigned task?
   - If the results are unexpected or incorrect, refine your query and re-run it.

3. Completion:
   - As soon as you obtain the query results, use the finalize() function to end your process.
   - In your final response, briefly highlight the key figures, facts, or insights you discovered.

Focus strictly on the assigned task, and finalize immediately after receiving the query results.
"""


class SQLAgent(Agent):
    def __init__(self, task, db_description, db_name, model_params):
        """
        Initialize the SQL Agent.
        """
        super().__init__(task=task, system_prompt=default_system_prompt, model_params=model_params, name="SQL Querist")
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
            self.turns += 1
            # Ensure the connection is closed
            if conn:
                conn.close()