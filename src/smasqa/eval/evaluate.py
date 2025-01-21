import pandas as pd
import random

batch_data_path = "path_to_data"
labeled_data = pd.read_csv(batch_data_path)

# Stub for the model


def model_stub(question, answers):
    """
    Model stub: Always selects the first answer.
    """
    return answers[0]


def evaluate_system(row):
    """Evaluate valuate the system with shuffled answers"""
    answers = [row["Answer 1"], row["Answer 2"],
               row["Answer 3"], row["Answer 4"]]
    shuffled_answers = random.sample(answers, len(answers))
    correct_answer = row["Answer 1"]
    model_choice = model_stub(row["question"], shuffled_answers)

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
