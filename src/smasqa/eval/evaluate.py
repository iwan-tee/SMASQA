from smasqa.agents.orchestrator import Orchestrator
import pandas as pd
from time import time
import sqlite3
LOG_FILE = "stat_data_results.csv"  # File to log evaluation results
MAX_RETRIES = 3  # Number of retries if Swarm fails


def get_db_description(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        return f"No tables found in {db_name}"

    db_description = []

    for table in tables:
        table_name = table[0]

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        table_desc = f"Table: {table_name}, Fields: {', '.join(column_names)}"
        db_description.append(table_desc)

    conn.close()

    return " | ".join(db_description)

def model_run(task, options, db_name="amazon.db",
              db_description='Table: amazon_sales, Fields: invoice_id, branch, city, customer_type, gender, product_line, unit_price, quantity, vat, total, date, time, payment_method, cogs, gross_margin_percentage, gross_income, rating, time_of_day, day_name, month_name'):
    """
    Runs Orchestrator with a given task and answer options.
    """
    orchestrator = Orchestrator(
        task=f"""Your task is: {task}
        Additional info that you'll need: 
        Database header row(description): {db_description}
        Database name: {db_name}""",
        options=options
    )
    return orchestrator.run()


def evaluate_row(row):
    """
    Evaluates whether Swarm selects the correct answer.
    Handles errors and retries failed attempts.
    """
    task = row['question']
    options = [f"Answer 1: {row['Answer 1']}",
               f"Answer 2: {row['Answer 2']}",
               f"Answer 3: {row['Answer 3']}",
               f"Answer 4: {row['Answer 4']}"
               ]
    db_name = row['file_name'].replace("csv", "db")
    db_description = get_db_description(db_name)

    result = None
    start_time = time()
    for attempt in range(MAX_RETRIES):
        try:
            result = model_run(task, options, db_name=db_name, db_description=db_description)
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
        f.write(f"{task};{result};{correct};{latency}\n")

    return correct


def evaluate_all(dataset):
    """
    Runs evaluation for all questions in the dataset, counting correct answers.
    """
    total = 0
    trues = 0

    # Open log file and write the header
    with open(LOG_FILE, "w") as f:
        f.write("Question;Answer Received;Is Correct;Time Taken\n")

    data = pd.read_csv(dataset, sep=';')

    for index, row in data.iterrows():
        print(f"Task #{index}")
        print(f"Task Description: {row['question']}\n")

        result = evaluate_row(row)
        if result:
            trues += 1
        total += 1

    print(f"Total: {total}, correct: {trues}")


# Run evaluation
evaluate_all("src/smasqa/eval/datasets/stat_data.csv")
