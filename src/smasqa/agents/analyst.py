import sqlite3
from typing import Tuple
from openai import OpenAI

from ..utils.repl import pretty_print_messages
from ..agents.agent import Agent

default_system_prompt = """
You are a business analyst. Your task is to analyze execution results to determine the correct answer to a given question.

### Instructions:
1. Carefully read the provided question and its four answer choices.
2. Analyze the provided data to extract key numerical insights, trends, and facts.
3. Compare the extracted insights with the answer choices.
4. Select the answer choice that best aligns with the data findings.
5. If the results are ambiguous, make an educated guess based on the available data.
6. Always justify your answer by summarizing the relevant insights.

### Example:
**Question:** What was the total revenue for Q4 2023?
**Answer Choices:** 
A) $12M 
B) $15M 
C) $18M 
D) $20M

**Data Analysis:** The provided dataset indicates that total revenue for Q4 2023 is $15M.

**Final Answer:** B) $15M
"""


class Analyst(Agent):
    def __init__(self, question, options, data) -> None:
        super().__init__(system_prompt=default_system_prompt, task=question)
        self.question = question
        self.options = options
        self.data = data
        self.functions = [self.choose_answer1, self.choose_answer2, self.choose_answer3, self.choose_answer4]

    def choose_answer1(self):
        return self.options[0]

    def choose_answer2(self):
        return self.options[1]

    def choose_answer3(self):
        return self.options[2]

    def choose_answer4(self):
        return self.options[3]

    def run(self) -> str:
        """
        Analyzes the data and determines the most accurate answer choice.
        """
        print("Running Analyst Agent with task:", self.question)
        user_message = f"user_query: {self.question}\n data: {self.data}"
        self.history.append({"role": "user", "content": user_message})
        self.agent_instance.functions = self.functions

        while not self.finished and len(self.history)-2 < self.max_turns:
            print(f"Analyzing... {len(self.history)}/{self.max_turns}")
            response = self.ai_env.run(agent=self.agent_instance, messages=self.history)
            pretty_print_messages(response.messages)
            if not self.finished:
                self.history.extend(response)

        return self.history[-1]["content"]



