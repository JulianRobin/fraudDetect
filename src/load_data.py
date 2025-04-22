
import pandas as pd

# Load dataset with error handling
data_path = "../data/creditcard.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: The file at {data_path} does not exist.")
    exit()

# Basic info
print("Dataset Info:")
print(data.info())
print(data.head())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())
