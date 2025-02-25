import traceback
from ..utils.repl import pretty_print_messages, process_and_print_streaming_response
from ..agents.agent import Agent

default_system_prompt = """
You are a Python coding agent specializing in data analysis.
You can work only with textual data as long as visualization is not supported in the environment.

Tasks:

1. Code Generation & Execution
    - Generate Python code for data analysis.
    - Execute the code and validate that it produces the expected results.

2. Critical Evaluation & Debugging
    - Assess whether the code works as intended.
    - If it fails, debug and adjust it.
    - Consider edge cases, data formats, and separator variations in pandas.

3. Finalization & Insights
    - Once the analysis is complete, summarize the key insights obtained.
    - Use finalize() to conclude the conversation.
"""


class CoderAgent(Agent):
    def __init__(self, task, datasets=[], streaming=False):
        super().__init__(
            system_prompt=default_system_prompt,
            task=task,
            functions=[self.run_code, self.finalize],
            agent_name="Coder Agent", streaming=streaming)

        self.datasets = datasets
        self.code_run_count = 0

    def get_available_datasets(self):
        """Return a list of available datasets."""
        return self.datasets

    def run_code(self, code: str) -> dict:
        """
        Executes the generated Python code.

        :param code: Python script to execute.
        :return: Result of execution saved in dict or an error message.
        """
        namespace = {}

        self.code_run_count += 1
        print(f"Running code - {self.code_run_count} iteration...")
        try:
            exec(code, {}, namespace)
            print("DEBUG MESSAGE: Code executed successfully!")
            return namespace
        except Exception as e:
            print(f"Execution error: {e}\n{traceback.format_exc()}")
            print("DEBUG MESSAGE: Code execution failed!")
            print(code)
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
        user_message = f"user_query: {self.task}\n Available datasets: {str(self.datasets)}\n"
        self.history.append({"role": "user", "content": user_message})
        self.agent_instance.functions = self.functions

        while not self.finished and len(self.history) - 2 < self.max_turns and self.code_run_count <= self.max_turns:
            print(f"Coding... {len(self.history) - 2}/{self.max_turns}")
            response = self.ai_env.run(
                agent=self.agent_instance, messages=self.history, stream=self.streaming)
            process_and_print_streaming_response(
                response) if self.streaming else pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        if (self.code_run_count == self.max_turns and self.finished == False):
            self.history.append(
                {"role": "assistant", "content": "Code execution limit reached with no valid results."})
            self.finished = True

        return self.history[-1]["content"]
