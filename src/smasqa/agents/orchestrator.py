from _ctypes import Structure

from ..agents.explorer import Explorer
from ..utils.repl import pretty_print_messages, process_and_print_streaming_response
from ..agents.agent import Agent
from ..agents.sql_agent import SQLAgent
from ..agents.coder import CoderAgent


class Orchestrator(Agent):
    def __init__(self, task, datasets, options=None, agent_calls_count=None):
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
                       self.transfer_to_coder_agent,
                       self.finalize,
                       self.return_answer,
                       self.get_available_datasets,
                       self.get_options],
            task=task,
            agent_name="Orchestrator",
            streaming=True
        )
        self.options = options
        self.datasets = datasets
        self.agent_calls_count = agent_calls_count

    def get_available_datasets(self):
        """
        Retrieve the list with paths (names) of available datasets.
        """
        return self.datasets

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

    def transfer_to_coder_agent(self, task, datasets=[]):
        """
        Formulate the task for the coder agent. If the task needs datasets, provide their full paths to the coder agent.

        :param task: Clearly formulated well structured task for a coder
        :param datasets [optional]: A list of datasets to transfer the task to.
        :return: code execution result in dict format or an error message.
        """
        self.agent_calls_count["coder"] += 1
        print(datasets)
        if datasets:
            coder = CoderAgent(task, datasets, streaming=True)
            return coder.run()

        coder = CoderAgent(task, streaming=True)
        return coder.run()

    def transfer_to_explorer(self, task: str, db_path: str):
        """
        Delegate structure extraction of a single a .db database or .csv dataset request to the Explorer agent.

        :param task: Clearly formulated request for explorer containing full name/path to the desired dataset.
        :param db_name: The full path to the database.
        :return: The database schema / dataset information in a dict structured format or an error message.
        """
        self.agent_calls_count["explorer"] += 1
        explorer = Explorer(task, db_path=db_path, model_params={
                            "model": "gpt-4o-mini"}, streaming=True)
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
        self.agent_calls_count["sql"] += 1
        sql_agent = SQLAgent(
            task=task,
            db_description=db_description,
            db_name=db_name,
            model_params={"model": "gpt-4o-mini"}, streaming=True
        )
        return sql_agent.run()

    def run(self) -> str:
        """
        Run the orchestrator.
        """
        print("Running Orchestrator...")
        print(self.task)

        user_query = f"user_query: {self.task}"
        answer_options = f"user predefined answer_options: {self.options}"
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "user", "content": answer_options})

        self.agent_instance.functions = self.functions
        response = self.ai_env.run(agent=self.agent_instance,
                                   messages=self.history, stream=self.streaming)
        process_and_print_streaming_response(
            response) if self.streaming else pretty_print_messages(response.messages)
        if not self.finished:
            self.history.extend(response)

        return self.history[-1]["content"]
