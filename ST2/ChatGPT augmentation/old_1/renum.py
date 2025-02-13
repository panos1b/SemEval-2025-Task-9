import pandas as pd

def renumber_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Check if the column "Unnamed: 0" exists
    if "Unnamed: 0" in df.columns:
        # Rename the "Unnamed: 0" column with a range from 0 to len(df) - 1
        df["Unnamed: 0"] = range(len(df))
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"File saved as {output_file}")
    else:
        print("The column 'Unnamed: 0' was not found in the CSV file.")

# Example usage:
input_file = 'combined_set.csv'  # Replace with your input file path
output_file = 'combined_set_renum.csv'  # Replace with your output file path
renumber_csv(input_file, output_file)
