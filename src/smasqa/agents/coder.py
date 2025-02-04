import traceback
from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
        You are a Python coding agent. Your job is to generate Python code for data analysis, execute it,
        and validate that it produces expected results. Think critically about whether the code works as intended.
        If it fails, debug and adjust it.
        When you're done, finalize() the conversation by summarizing the insights obtained from the analysis.
"""


class CoderAgent(Agent):
    def __init__(self, task, dataset_path):
        super().__init__(task=task, system_prompt=default_system_prompt)
        self.dataset_path = dataset_path
        self.functions = [self.run_code, self.finalize]

    def run_code(self, code: str) -> str:
        """
        Executes the generated Python code.

        :param code: Python script to execute.
        :return: Result of execution or error message.
        """
        try:
            exec(code)
            return "Code executed successfully."
        except Exception as e:
            return f"Execution error: {e}\n{traceback.format_exc()}"

    def finalize(self, results) -> None:
        """
        Finalizes the conversation.

        :param results: Final results of code execution.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True
        print("Final result:", results)

    def run(self) -> str:
        """
        Generates Python code for data analysis and runs it.
        """
        print("Running CoderAgent with task:", self.task)
        user_message = f"user_query: {self.task}\n dataset path: {self.dataset_path}"
        self.history.append({"role": "user", "content": user_message})
        self.agent_instance.functions = self.functions

        while not self.finished and len(self.history) - 2 < self.max_turns:
            print(f"Coding... {len(self.history)}/{self.max_turns}")
            response = self.ai_env.run(agent=self.agent_instance, messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]
