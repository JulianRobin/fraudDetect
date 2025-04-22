
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = "../data/creditcard.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: The file at {data_path} does not exist.")
    exit()

# Basic dataset information
print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Check for class distribution
class_counts = data['Class'].value_counts()
print("\nClass Distribution:")
print(class_counts)

# Plot class distribution with counts and log scale
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.yscale('log')  # Use log scale for better visibility
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Number of Transactions (log scale)")

# Annotate counts on the bars
for i, count in enumerate(class_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)
