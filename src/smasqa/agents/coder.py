import traceback
from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are a Python coding agent specializing in data analysis.

Tasks:

1. Code Generation & Execution
    - Generate Python code for data analysis.
    - Execute the code and validate that it produces the expected results.

2. Critical Evaluation & Debugging
    - Assess whether the code works as intended.
    - If it fails, debug and adjust it.
    - Consider edge cases, data formats, and separator variations in pandas.
    - The environment does not support inline plots. So, save them to the folder "src/smasqa/eval/datasets/plots/" as images.

3. Finalization & Insights
    - Once the analysis is complete, summarize the key insights obtained.
    - Use finalize() to conclude the conversation.
"""


class CoderAgent(Agent):
    def __init__(self, task, datasets=[]):
        super().__init__(
            system_prompt=default_system_prompt,
            task=task,
            functions=[self.run_code, self.finalize, self.get_available_datasets],
            name="Python Coder"
        )
        self.datasets = datasets

    def get_available_datasets(self):
        """Return a list of available datasets."""
        self.turns += 1
        return self.datasets


    def run_code(self, code: str) -> dict:
        """
        Executes the generated Python code.

        :param code: Python script to execute.
        :return: Result of execution saved in dict or an error message.
        """
        namespace = {}

        try:
            exec(code, {}, namespace)
            return namespace
        except Exception as e:
            return f"Execution error: {e}\n{traceback.format_exc()}"
        finally:
            self.turns += 1