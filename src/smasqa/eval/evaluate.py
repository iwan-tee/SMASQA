from smasqa.agents.orchestrator import Orchestrator
import pandas as pd
from time import time
import sqlite3
import json
import swarm
import tiktoken
import openai
import random

LOG_FILE = "src/smasqa/eval/results/3 agents - all gpt-4o/merged_results.csv"  # File to log evaluation results
MAX_RETRIES = 3  # Number of retries if Swarm fails

inputTokenCount = 0
outputTokenCount = 0
totalTokenCount = 0

original_get_chat_completion = swarm.Swarm.get_chat_completion

# Choose an encoding based on the model used (GPT-4, GPT-3.5, etc.)
encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    """Helper function to count tokens in a given text."""
    return len(encoding.encode(text))


def estimate_tokens(messages: list) -> int:
    """Estimate total tokens in a list of messages."""
    return sum(count_tokens(json.dumps(msg)) for msg in messages)

def get_chat_completion_with_token_count(self, agent, history, context_variables, model_override, stream, debug):
    """
    Wrapped function that counts input and output tokens while calling the original function.
    """
    global inputTokenCount
    global outputTokenCount
    global totalTokenCount

    # Call the original function
    response = original_get_chat_completion(
        self, agent, history, context_variables, model_override, stream, debug)

    output_tokens, input_tokens, total_tokens = 0, 0, 0

    if response.usage is not None:
        output_tokens = response.usage.completion_tokens
        input_tokens = response.usage.prompt_tokens
        total_tokens = response.usage.total_tokens

    outputTokenCount += output_tokens
    inputTokenCount += input_tokens
    totalTokenCount += total_tokens

    return response


# Monkey patch the method
swarm.Swarm.get_chat_completion = get_chat_completion_with_token_count


def model_run(task, options, db_name):
    """
    Runs Orchestrator with a given task and answer options.
    """
    orchestrator = Orchestrator(
        task=f"Your task is: {task}",
        datasets=db_name,
        options=options
    )
    result = orchestrator.run()[0]
    orchestrator_turns = orchestrator.turns
    servant_turns = orchestrator.servant_turns
    return result, orchestrator_turns, servant_turns

def evaluate_row(row):
    """
    Evaluates whether Swarm selects the correct answer.
    Handles errors and retries failed attempts.
    """
    global totalTokenCount
    global inputTokenCount
    global outputTokenCount

    task_id = row['ID']
    task = row["Question"]
    db_name = row['db_path']
    level = row['level']

    original_answers = [
        ("Answer 1", row['Answer 1']),
        ("Answer 2", row['Answer 2']),
        ("Answer 3", row['Answer 3']),
        ("Answer 4", row['Answer 4'])
    ]

    shuffled_answers = original_answers.copy()
    shuffle_map = dict()
    random.shuffle(shuffled_answers)
    for i in range(4):
        shuffle_map[f'Answer {i+1}'] = shuffled_answers[i][0]

    print(shuffle_map)
    print(shuffled_answers)

    # Формируем новый список для передачи в модель
    options = [f'Answer {i+1}: {shuffled_answers[i][1]}' for i in range(4)]



    print(options)

    result = None
    start_time = time()
    for attempt in range(MAX_RETRIES):
        try:
            results_all = model_run(task, options, db_name=db_name)
            selected_option = results_all[0]
            result = shuffle_map[selected_option]
            orchestrator_turns = results_all[1]
            SQL_Turns = results_all[2]['SQLAgent']
            Coder_Turns = results_all[2]['CoderAgent']
            Explorer_Turns = results_all[2]['Explorer']
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Swarm error (attempt {attempt + 1}): {e}")
    # If Swarm fails after all retries, consider it incorrect
    if result is None:
        result = "ERROR"
    end_time = time()
    latency = end_time - start_time
    correct = (result == "Answer 1")

    # Log results in CSV
    with open(LOG_FILE, "a") as f:
        f.write(f"{task_id};{task};{level};{result};{correct};{latency};{orchestrator_turns};{Explorer_Turns};{SQL_Turns};{Coder_Turns};{inputTokenCount};{outputTokenCount};{totalTokenCount}\n")

    inputTokenCount = 0
    outputTokenCount = 0
    totalTokenCount = 0

    return correct


def evaluate_all(dataset):
    """
    Runs evaluation for all questions in the dataset, counting correct answers.
    """
    total = 0
    trues = 0

    # Open log file and write the header
    with open(LOG_FILE, "w") as f:
        f.write("Question ID;Question;Difficulty Level;Model Output;IsCorrect;Latency;Orchestrator_Turns;Explorer_Turns;SQLAgent_Turns;Coder_Turns;Input Tokens;Output Tokens;Total Tokens\n")

    data = pd.read_csv(dataset, sep=';')
    for index, row in data.iterrows():
        print(f"Task #{index}")
        print(f"Task Description: {row['Question']}\n")

        result = evaluate_row(row)
        if result:
            trues += 1
        total += 1

    print(f"Total: {total}, correct: {trues}")


# Run evaluation
evaluate_all("src/smasqa/eval/datasets/questions_merged_.csv")