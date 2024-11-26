from openai import OpenAI

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

        completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                    ]
                )

        sql_query = completion.choices[0].message.content
        return sql_query
