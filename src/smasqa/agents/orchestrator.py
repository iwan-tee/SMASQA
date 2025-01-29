from smasqa.utils.repl import pretty_print_messages
from smasqa.agents.agent import Agent
from smasqa.agents.sql_agent import SQLAgent


class Orchestrator(Agent):
    def __init__(self, task):
        super().__init__(
            system_prompt="""You are an orchestrator. Your job is to solve complex tasks of a user using agents available to you.
              To solve users' tasks follow these steps:
              1. Create and write out a detailed plan to solve the task.
              2. Review the plan and identify which agents would be needed.
              3. Formulate the task description for the agent. Be clear, explicit and exhaustive.
              4. Transfer the task to the appropriate agent using one of the transfer functions like transfer_to_sql_agent() or transfer_to_explorer().
              5. Analyse the response from the agent and decide the next steps.
              6. Repeat steps 3-5 until the task is completed.
                7. When the task is completed use finalize() to end the conversation and provide results to the user.
              """,
            functions=[self.transfer_to_sql_agent, self.finalize],
            task=task
        )

    def finalize(self, results) -> None:
        """
        Finalizes the conversation.

        :param results: Consolidated results of executing sql query.
        """
        # self.cb(results)
        self.history.append({"role": "assistant", "content": results})
        self.finished = True
        print(self.history[-1])

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
        user_message = f"user_query: {self.task}"
        self.history.append({"role": "user", "content": user_message})

        self.agent_instance.functions = self.functions
        response = self.ai_env.run(agent=self.agent_instance,
                                   messages=self.history)
        pretty_print_messages(response.messages)
        if not self.finished:
            self.history.extend(response)

        return self.history[-1]["content"]
