import pandas as pd
import random
from openai import OpenAI

default_file_name = "batch_1_enriched.csv"
default_path = "./src/smasqa/eval/datasets/"
labeled_data = pd.read_csv(default_path + default_file_name, ";")


def model_stub(question, answers, dataset=None):
    """
    Model stub: Always selects the first answer.
    """
    return answers[0]


def model_run(question, answers, dataset=None):
    """
    Chooses the best answer from an array of answers based on a dataset using OpenAI API.

    Parameters:
    - question (str): The question asked.
    - answers (list of str): List of possible answers.
    - dataset (pd.DataFrame): Dataset relevant to the question.

    Returns:
    - str: The chosen answer.
    """
    # Convert dataset to a textual summary
    dataset_text = " ".join(dataset.astype(str).fillna("NA").values.flatten())

    # Prepare the messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": f"Question: {question}\nDataset: {dataset_text}\nAnswers: {', '.join(answers)}\n\nBased on the dataset and the question, choose the most appropriate answer from the provided options and explain your reasoning."}
    ]

    try:
        # Initialize OpenAI client
        client = OpenAI()

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # Extract the answer from the response
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        return f"Error in processing: {str(e)}"


def evaluate_system(row):
    """Evaluate valuate the system with shuffled answers"""
    answers = [row["Answer 1"], row["Answer 2"],
               row["Answer 3"], row["Answer 4"]]
    shuffled_answers = random.sample(answers, len(answers))
    correct_answer = row["Answer 1"]

    if 'file_name' not in row:
        row['file_name'] = default_file_name

    data = pd.read_csv("./src/smasqa/eval/datasets/" + row['file_name'], ";")

    model_choice = model_stub(row["question"], shuffled_answers, data)

    is_correct = model_choice == correct_answer
    return is_correct, correct_answer, model_choice


# Run the evaluation
results = labeled_data.apply(evaluate_system, axis=1, result_type="expand")
results.columns = ["is_correct", "correct_answer", "model_choice"]

# Concatenate results with the original dataset
labeled_data = pd.concat([labeled_data, results], axis=1)

# Calculate metrics
total = len(labeled_data)
true_positives = labeled_data["is_correct"].sum()
false_positives = (labeled_data["model_choice"]
                   != labeled_data["correct_answer"]).sum()
false_negatives = total - true_positives

precision = true_positives / \
    (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / \
    (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = (2 * precision * recall) / (precision +
                                       recall) if (precision + recall) > 0 else 0

# Output metrics
print(f"Accuracy: {true_positives / total:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1_score:.2%}")
