import sqlite3
from typing import Tuple
from openai import OpenAI

from smasqa.utils.repl import pretty_print_messages
from smasqa.agents.agent import Agent

default_system_prompt = """
        You are an SQL agent. Your job is to generate SQL queries and run them. 
        You should create a sqlite query, execute it using run_query(). Think if the code is working as expected. Tweak it if it's not producing expected result.
        When you are done with the query and satisfied with the results use finalize() to end the conversation and provide summarized results focusing on numbers and facts that you've discovered."""


class SQLAgent(Agent):
    def __init__(self, task, db_description, db_name):
        """
        Initialize the SQL Agent.
        """
        super().__init__(task=task, system_prompt=default_system_prompt)
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
        database_path = f"src/smasqa/eval/datasets/db/{self.db_name}"

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
        Finalizes the conversation.

        :param results: Consolidated results of executing sql query.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True

    def run(self) -> str:
        """
        Generate an SQL query based on the user's natural language query and tests if it runs.
        """
        print("Running SQL agent with the following task:", self.task)
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


class TextToSqlAgent:
    def __init__(self, db_description):
        """
        Initialize the Text2SQl Agent.
        """
        self.client = OpenAI()
        self.db_description = db_description

    def generate(self, user_query):
        system_message = f"""
        You are an AI assistant specialized in generating SQL queries.
        Your task is to take natural language requests and convert them into accurate and efficient SQL statements.

        Follow these instructions carefully:
        1. Understand the user's intent and the structure of the data they describe.
        2. Generate SQL queries tailored to the request.
        3. Use standard SQL syntax unless specified otherwise by the user.
        4. If the user specifies a database system (e.g., MySQL, PostgreSQL, SQLite), adapt the query accordingly.
        5. Assume the user provides table and column names as they are, unless otherwise noted.

        Provide the SQL query as your response, without extra explanations unless requested.
        Example:
        User Query: "List all customers from the customers table who live in California."
        SQL Code: SELECT * FROM customers WHERE state = 'California'

        INPUT:
        1. user_query: natural language query from the user
        2. database_description: optional: create table statements or text description

        OUTPUT:
        sql-query
            Return it as a string with no leading apostrophes or smth like that
        """
        user_message = f"user_query: {user_query}\n database description: {self.db_description}"

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        sql_query = completion.choices[0].message.content
        return sql_query
