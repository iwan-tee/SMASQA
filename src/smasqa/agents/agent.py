from swarm import Swarm, Agent as SwarmAgent
from functools import wraps
from ..utils.repl import pretty_print_messages

model_params = {
    "model": "gpt-4o",
}


class Agent:
    def __init__(self, system_prompt, task, name="Agent", model_params=model_params, functions=[]):
        """
        Initialize the Agent.
        """
        self.ai_env = Swarm()
        self.agent_instance = SwarmAgent(model=model_params.get("model"))
        self.system_prompt = system_prompt
        self.task = task
        self.model_params = model_params
        self.history = [{"role": "system", "content": system_prompt}, {
            "role": "user", "content": task}]
        self.max_turns = 10
        self.finished = False
        self.functions = functions
        self.turns = 0
        self.agent_instance.name = name

    def run(self):
        """Runs the pipeline."""
        self.history.append({"role": "user", "content": f"Task is {self.task}"})
        self.agent_instance.functions = self.functions
        while not self.finished:
            response = self.ai_env.run(agent=self.agent_instance,
                                       messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response.messages)

        return self.history[-1]["content"], self.history, self.turns

    def finalize(self, results):
        """
        Finalize the task completion with the results.

        :param results: the results of the task.
        """
        self.history.append({"role": "assistant", "content": results})
        self.finished = True
