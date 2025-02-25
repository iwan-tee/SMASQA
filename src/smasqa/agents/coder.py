from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent
import sys
import io
import traceback


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
    def __init__(self, task, datasets=[]):
        super().__init__(
            system_prompt=default_system_prompt,
            task=task,
            functions=[self.run_code, self.finalize])
        self.datasets = datasets

    def get_available_datasets(self):
        """Return a list of available datasets."""
        return self.datasets

    def run_code(self, code: str):
        """
        Executes the generated Python code safely.

        :param code: Python script to execute.
        :return: Dict with execution results or error message.
        """
        print(f'DEBUG MSG: CODE IS\n{code}')
        namespace = {}

        # Перехват stdout и stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            exec(code, {}, namespace)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            # Фильтруем namespace
            clean_namespace = {k: v for k, v in namespace.items() if not k.startswith("__")}
            if not clean_namespace:
                clean_namespace["info"] = "Code executed, but no output variables found."

            return {
                "status": "success",
                "output": clean_namespace,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            }
        except Exception as e:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            }

    def finalize(self, results) -> None:
        """
        Finalizes the conversation.

        :param results: result of the run_code() on the code
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True

    def run(self) -> str:
        """
        Generates Python code for data analysis and runs it.
        """
        print("Running CoderAgent with task:", self.task)
        user_message = f"user_query: {self.task}\n"
        self.history.append({"role": "user", "content": user_message})
        self.agent_instance.functions = self.functions

        while not self.finished and len(self.history) - 2 < self.max_turns:
            print(f"Coding... {len(self.history)}/{self.max_turns}")
            response = self.ai_env.run(agent=self.agent_instance, messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]
