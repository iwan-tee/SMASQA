from smasqa.agents.orchestrator import Orchestrator
import pandas as pd
from time import time
import sqlite3
import json
import swarm
import tiktoken

LOG_FILE = "src/smasqa/eval/results/3 agents - all gpt-4o/merged_results.csv"  # File to log evaluation results
MAX_RETRIES = 3  # Number of retries if Swarm fails

inputTokenCount = 0
INPUTTOKENCOUNT = 0
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

    # Estimate input tokens
    input_tokens = estimate_tokens(history)
    print(f"Input Tokens: {input_tokens}")
    inputTokenCount = inputTokenCount + input_tokens

    # Call the original function
    response = original_get_chat_completion(
        self, agent, history, context_variables, model_override, stream, debug)

    # Estimate output tokens from `content`, `tool_calls`, `function_call`
    output_tokens = 0

    if isinstance(response, swarm.core.ChatCompletionMessage):
        if response.content:
            output_tokens += count_tokens(response.content)
        if response.function_call:
            output_tokens += count_tokens(
                json.dumps(response.function_call.dict()))
        if response.tool_calls:
            output_tokens += sum(count_tokens(json.dumps(tool_call.dict()))
                                 for tool_call in response.tool_calls)

    print(f"Output Tokens: {output_tokens}")
    outputTokenCount = outputTokenCount + output_tokens
    totalTokenCount = totalTokenCount + inputTokenCount + outputTokenCount

    return response


# Monkey patch the method
swarm.Swarm.get_chat_completion = get_chat_completion_with_token_count


def model_run(task, options, db_name):
    """
    Runs Orchestrator with a given task and answer options.
    """
    orchestrator = Orchestrator(
        task=f"Your task is: {task} \n Target Database: {db_name}",
        database=db_name,
        options=options
    )
    return orchestrator.run()


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
    db_name = row['db_path'].replace("csv", "db")
    level = row['level']

    options = [f"Answer 1: {row['Answer 1']}",
               f"Answer 2: {row['Answer 2']}",
               f"Answer 3: {row['Answer 3']}",
               f"Answer 4: {row['Answer 4']}"
               ]

    result = None
    start_time = time()
    for attempt in range(MAX_RETRIES):
        try:
            result = model_run(task, options, db_name=db_name)
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Swarm error (attempt {attempt + 1}): {e}")
    # If Swarm fails after all retries, consider it incorrect
    if result is None:
        result = "ERROR"
    else:
        result = result[:8]
    end_time = time()
    latency = end_time - start_time

    correct = (result == "Answer 1")

    # Log results in CSV
    with open(LOG_FILE, "a") as f:
        f.write(f"{task_id};{task};{level};{result};{correct};{latency};{inputTokenCount};{outputTokenCount};{totalTokenCount}\n")

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
        f.write("Question ID; Question; Difficulty Level; Model Output; IsCorrect; Latency; Input Tokens;Output Tokens;Total Tokens\n")

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
evaluate_all("src/smasqa/eval/datasets/questions_merged.csv")