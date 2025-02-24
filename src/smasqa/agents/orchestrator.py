from _ctypes import Structure

from ..agents.explorer import Explorer
from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent
from ..agents.sql_agent import SQLAgent


class Orchestrator(Agent):
    def __init__(self, task, database, options=None):
        super().__init__(
            system_prompt="""
            You are an orchestrator.
            Your job is to solve complex tasks of a user using agents available to you.
            
            To solve users' tasks follow these steps:
              1. Create and write out a detailed plan to solve the task.
              2. Review the plan and identify which agents would be needed.
              3. Formulate the task description for the agent. Be clear, explicit and exhaustive.
              4. Transfer the task to the appropriate agent using one of the transfer functions.
              5. Analyse the response from the agent and decide the next steps.
              6. Repeat steps 3-5 until the task is completed.
              7. When the task is completed 
                7.1 Use finalize() to end the conversation and provide results to the user
                7.2 User will provide you a list of four possible answers and one of them is correct. Format of the list is ["Answer 1: ...", ...]
                  As a final result provide just a string in format "Answer i"
              """,
            functions=[self.transfer_to_sql_agent, self.transfer_to_explorer,
                       self.finalize,
                       self.return_answer,
                       self.get_options],
            task=task
        )
        self.options = options
        self.database = database

    def get_options(self) -> list:
        """
        Retrieve the list of answer options provided by the user.

        :return: A list containing four answer choices.
        """
        return self.options

    def return_answer(self, option: int) -> str:
        """
        Return the selected answer based on the provided option index.

        :param option: An integer representing the answer choice (1 to 4).
        :return: A string in the format "Answer i" if valid, otherwise an error message.
        """
        if isinstance(option, int) and 1 <= option <= 4:
            return f"Answer {option}"
        return "Error! None of the answers are correct"

    def finalize(self, results: str) -> None:
        """
        Finalizes the conversation by appending the final result and marking completion.

        :param results: The answer option selected.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True
        print(self.history[-1])

    def transfer_to_explorer(self, task: str):
        """
        Delegate a database structure extraction request to the Explorer agent.

        :param task: A description of the database exploration task.
        :return: The database schema information in a dict structured format.
        """
        print(task)
        explorer = Explorer(task)
        structure = explorer.run()
        if isinstance(structure, dict):
            return structure
        structure = structure.replace('"""', "").replace("json\n", "").strip()
        return eval(structure)

    def transfer_to_sql_agent(self,
                              task,
                              db_description,
                              db_name) -> str:
        """
        Transfer the task, database description and database name to the SQLAgent.

        :param task: A natural language query or description of the SQL task.
        :param db_description: A dict structured description of the database schema.
        :param db_name: The name (path) of the database.

        :return: The query result as a string.
        """
        print(task, db_description, db_name)
        sql_agent = SQLAgent(
            task=task,
            db_description=db_description,
            db_name=db_name
        )
        return sql_agent.run()

    def run(self) -> str:
        """
        Run the orchestrator.
        """
        print("Running Orchestrator...")
        user_query = f"user_query: {self.task}"
        answer_options = f"user predefined answer_options: {self.options}"
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "user", "content": answer_options})

        self.agent_instance.functions = self.functions
        response = self.ai_env.run(agent=self.agent_instance,
                                   messages=self.history)
        pretty_print_messages(response.messages)
        if not self.finished:
            self.history.extend(response)

        return self.history[-1]["content"]
