from swarm import Swarm, Agent as SwarmAgent
from functools import wraps


model_params = {
    "model": "gpt-4o",
}


class Agent:
    def __init__(self, system_prompt, task, model_params=model_params, functions=[]):
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

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and name not in {"__init__", "__getattribute__", "run"}:
            @wraps(attr)
            def wrapped_method(*args, **kwargs):
                object.__setattr__(self, "turns", self.turns + 1)
                return attr(*args, **kwargs)
            return wrapped_method
        return attr

    def run():
        """Runs the pipeline."""

    def finalize():
        """Finalizes the conversation."""
