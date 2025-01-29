import sqlite3
import pandas as pd
import os


def csv_to_sqlite(dataset_csv_path, output_dir, path_prefix="src/smasqa/eval/datasets/raw_dbs/"):
    """
    Reads a dataset CSV file and converts all referenced <file_name>.csv files into SQLite databases.

    Parameters:
        dataset_csv_path (str): Path to the dataset CSV file containing file_name column.
        output_dir (str): Directory to save the generated SQLite database files.
    """
    if not os.path.exists(dataset_csv_path):
        print(f"Error: The dataset file {dataset_csv_path} does not exist.")
        return

    # Read the dataset CSV file
    try:
        dataset_df = pd.read_csv(dataset_csv_path, sep=';')
    except Exception as e:
        print(f"Error reading dataset CSV file: {e}")
        return

    if 'file_name' not in dataset_df.columns:
        print("Error: 'file_name' column not found in the dataset.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all unique file names in the 'file_name' column
    for file_name in dataset_df['file_name'].dropna().unique():
        # Assuming file_name contains the full path or relative path to the CSV file
        csv_file_path = path_prefix + file_name

        if not os.path.exists(csv_file_path):
            print(
                f"Warning: The file {csv_file_path} does not exist. Skipping.")
            continue

        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(csv_file_path, sep=',')
        except Exception as e:
            print(f"Error reading CSV file {csv_file_path}: {e}")
            continue

        # Extract the database name from the file name
        db_name = os.path.join(
            output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}.db")

        # Create or connect to the SQLite database
        try:
            conn = sqlite3.connect(db_name)
            print(f"Connected to SQLite database: {db_name}")
        except sqlite3.Error as e:
            print(f"Error creating database {db_name}: {e}")
            continue

        try:
            # Write the DataFrame to the SQLite database
            table_name = os.path.splitext(os.path.basename(file_name))[0]
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(
                f"Data from {csv_file_path} has been inserted into table '{table_name}' in {db_name}.")
        except Exception as e:
            print(f"Error inserting data into SQLite database {db_name}: {e}")
        finally:
            # Close the connection
            conn.close()
            print(f"SQLite connection for {db_name} closed.")


# Example usage
if __name__ == "__main__":
    # Replace with your dataset CSV file path
    dataset_csv = "src/smasqa/eval/datasets/business_data_last_version.csv"
    # Replace with your desired output directory for SQLite databases
    output_directory = "src/smasqa/eval/datasets/db"
    print(f"Executing from {os.getcwd()}")
    csv_to_sqlite(dataset_csv, output_directory)
