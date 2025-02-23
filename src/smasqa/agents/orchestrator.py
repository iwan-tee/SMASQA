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
              4. Transfer the task to the appropriate agent using one of the transfer functions like transfer_to_sql_agent() or transfer_to_explorer().
              5. Analyse the response from the agent and decide the next steps.
              6. Repeat steps 3-5 until the task is completed.
              7. When the task is completed 
                7.1 Use finalize() to end the conversation and provide results to the user
                7.2 User will provide you a list of four possible answers and one of them is correct. Format of the list is ["Answer 1: ...", ...]
                  As a final result provide just a string in format "Answer i"
              """,
            functions=[self.transfer_to_sql_agent, self.transfer_explorer,
                       self.finalize,
                       self.return_answer1, self.return_answer2,
                       self.return_answer3, self.return_answer4,
                       self.return_answer_error,
                       self.get_db_name, self.get_options],
            task=task
        )
        self.options = options
        self.database = database

    def get_db_name(self):
        """
        Return the name (path) of the target database
        """
        return self.database

    def get_options(self):
        """
        Return the user provided options.
        """
        return self.options

    def return_answer1(self):
        """
        Return the answer 1
        """
        return "Answer 1"

    def return_answer2(self):
        """
        Return the answer 2
        """
        return "Answer 2"

    def return_answer3(self):
        """
        Return the answer 3
        """
        return "Answer 3"
    def return_answer4(self):
        """
        Return the answer 4
        """
        return "Answer 4"

    def return_answer_error(self):
        """
        Return the Error if none of the user provided answer options are correct.
        """
        return "Error occurred! None"


    def finalize(self, results) -> None:
        """
        Finalizes the conversation.

        :param results: Consolidated results of executing sql query.
        """
        # self.cb(results)
        self.history.append({"role": "assistant", "content": results})
        self.finished = True
        print(self.history[-1])

    def transfer_explorer(self, task):
        """
        Transfer the task to explore the database structure to the Explorer agent
        :param task: Database exploration request.
        """
        explorer = Explorer(task)
        return explorer.run()


    def transfer_to_sql_agent(self, task, db_description, db_name):
        """Transfer to sql agent when you need to generate a SQL query.
        :param task: Task description.
        :param history: Conversation history.
        :param db_description: Database schema description.
        :param db_name: Database name."""

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
