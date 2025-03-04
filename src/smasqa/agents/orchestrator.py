from ..agents.explorer import Explorer
from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent
from ..agents.sql_agent import SQLAgent
from ..agents.coder import CoderAgent


class Orchestrator(Agent):
    def __init__(self, task, datasets, options=None):
        super().__init__(
            system_prompt="""
            # Your role
            You are an orchestrator in the Multi-agentic system.
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
            task=task
        )
        self.options = options
        self.datasets = [f"src/smasqa/eval/datasets/db/{x}" if x.endswith(
            ".db") else f"src/smasqa/eval/datasets/raw_dbs/{x}" for x in eval(datasets)]


    def get_available_datasets(self):
        """
        Retrieve the list with paths of available datasets.
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
        Finalize the conversation by appending the final result and marking completion.

        :param results: The answer option selected.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True

    def transfer_to_coder_agent(self, task, datasets=[]):
        """
        Formulate the task for the coder agent. If the task needs datasets, provide their full paths to the coder agent.

        :param task: Clearly formulated well structured task for a coder
        :param datasets [optional]: A list of datasets to transfer the task to.
        :return: code execution result in dict format or an error message.
        """
        if datasets:
            return CoderAgent(task, datasets).run()
        return CoderAgent(task).run()

    def transfer_to_explorer(self, task: str):
        """
        Delegate structure extraction of a single a .db database or .csv dataset request to the Explorer agent.

        :param task: Clearly formulated request for explorer containing full name/path to the desired dataset.
        :return: The database schema / dataset information in a dict structured format or an error message.
        """
        explorer = Explorer(task, model_params={"model": "gpt-4o-mini"})
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

        sql_agent = SQLAgent(
            task=task,
            db_description=db_description,
            db_name=db_name,
            model_params={"model": "gpt-4o-mini"}
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
        while not self.finished and len(self.history) - 2 < self.max_turns:
            response = self.ai_env.run(agent=self.agent_instance,
                                       messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]
