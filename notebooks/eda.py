import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="darkgrid")

# Load dataset
df = pd.read_csv("data/CICIDS2017_cleaned.csv")

# Show first few rows
print("\nFirst few rows of the dataset:\n")
print(df.head())

# Dataset info
print("\nDataset info:\n")
print(df.info())

# Missing values
print("\nMissing values:\n")
print(df.isnull().sum())

# Label distribution
print("\nLabel distribution:\n")
print(df['Attack Type'].value_counts())

# Plot label distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Attack Type', data=df)
plt.xticks(rotation=45)
plt.title("Distribution of Normal vs Attack Traffic")
plt.tight_layout()
plt.savefig("notebooks/label_distribution.png")  # Saves the plot
plt.show()

# Basic statistics
print("\nStatistical summary:\n")
print(df.describe())
