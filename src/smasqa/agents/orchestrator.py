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
            You are an orchestrator in a Multi-agentic system.
            Your job is to solve complex tasks of a user using agents available to you.            
                
            To solve users' tasks follow these steps:
              1. Create and write out a detailed plan to solve the task.
              2. Review the plan and identify which agents would be needed.
              3. Formulate the task description for the agent. Be clear, explicit and exhaustive.
              4. Transfer the task to the appropriate agent using one of the transfer functions.
              5. Analyse the response from the agent and decide the next steps.
              6. Repeat steps 3-5 until the task is completed.
              7. When the task is completed, use finalize() to end the conversation and provide results to the user
              """,
            functions=[self.transfer_to_sql_agent, self.transfer_to_explorer,
                       self.transfer_to_coder_agent,
                       self.finalize,
                       self.get_available_datasets],
            task=task,
            name="Orchestrator"
        )
        self.options = options
        self.datasets = [f"src/smasqa/eval/datasets/db/{x}" if x.endswith(
            ".db") else f"src/smasqa/eval/datasets/raw_dbs/{x}" for x in eval(datasets)]

        self.servant_turns = {
            "SQLAgent": 0,
            "CoderAgent": 0,
            "Explorer": 0
        }



    def get_available_datasets(self):
        """
        Retrieve the list with paths of available datasets.
        """
        self.turns += 1
        return self.datasets

    def get_options(self) -> list:
        """
        Retrieve the list of answer options provided by the user.

        :return: A list containing four answer choices.
        """
        self.turns += 1
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


    def transfer_to_coder_agent(self, task, datasets=[]):
        """
        Formulate the task for the coder agent. If the task needs datasets, provide their full paths to the coder agent.

        :param task: Clearly formulated well structured task for a coder
        :param datasets [optional]: A list of datasets to transfer the task to.
        :return: code execution result in dict format or an error message.
        """
        if datasets:
            coder = CoderAgent(task, datasets)
            return coder.run()
        coder = CoderAgent(task)
        self.turns += 1
        results = coder.run()
        self.servant_turns["CoderAgent"] += results[2]
        return results[0]

    def transfer_to_explorer(self, task: str):
        """
        Delegate structure extraction of a single a .db database or .csv dataset request to the Explorer agent.

        :param task: Clearly formulated request for explorer containing full name/path to the desired dataset.
        :return: The database schema / dataset information in a dict structured format or an error message.
        """
        self.turns += 1
        explorer = Explorer(task, model_params={"model": "gpt-4o-mini"})
        results = explorer.run()
        self.servant_turns["Explorer"] += results[2]
        structure = results[0]
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
        self.turns += 1
        sql_agent = SQLAgent(
            task=task,
            db_description=db_description,
            db_name=db_name,
            model_params={"model": "gpt-4o-mini"}
        )
        results = sql_agent.run()
        self.servant_turns["SQLAgent"] += results[2]
        return sql_agent.run()[0]