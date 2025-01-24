import pandas as pd
import os

def merge1(file1_path, file2_path, save_dir_path):
    # Read the CSV files into DataFrames
    file1_data = pd.read_csv(file1_path)
    file2_data = pd.read_csv(file2_path)

    # Append weather elements (without mean and std dev) to File 1
    for _, row in file2_data.iterrows():
        element = row['Weather Element']  # Use the weather element name as the column name
        if element not in file1_data.columns:  # Check to avoid overwriting columns
            file1_data[element] = row['Data']  # Append the data directly from File 2

    # File path for saving the combined data
    output_csv_path = os.path.join(save_dir_path, "combined_data_no_mean_std.csv")
    file1_data.to_csv(output_csv_path, index=False)

    print(f"Combined data saved to {output_csv_path}")

    return output_csv_path
