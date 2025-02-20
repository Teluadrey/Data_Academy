import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
There I imported the necessary libraries for the analysis. You can install the libraries using the following command(write it in your therminal, also, do not forget to create a virtual environment before installing those libraries: 
pip install pandas numpy matplotlib seaborn
"""

"""
First, we need to load the data into a DataFrame using the read_csv function from the pandas library.
"""
# Load the data
data_path = "../Data_Academy/accident.csv"  # Path to the CSV file containing the data.
# There you can notice that I used ../, I used it because I did not really want to mention the full path of file(it would take a lot of time), so I just mentioned the path of the file from the current directory.

df = pd.read_csv(data_path)  # Read the CSV file into a DataFrame

df.head()  # Display the first few rows of the DataFrame

# Data Visualization

# See whether the gender of the driver has an impact on the survival
plt.figure(figsize=(10, 6))  # Set the figure size
sns.countplot(
    x="Survived", hue="Gender", data=df
)  # Create a count plot for survival by gender
plt.title("Survival Count by Gender")  # Set the title of the plot
plt.xlabel("Survived")  # Set the x-axis label(text for x axis)
plt.ylabel("Count")  # Set the y-axis label(text for y axis)
plt.legend(
    title="Gender"
)  # Set the legend title, it will be displayed on the right top of the plot
plt.show()  # Display the plot

# See whether the age of the driver has an impact on the survival
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(
    df["Age"], bins=30, kde=True
)  # Create a histogram for age distribution with KDE(kernel density estimation) enabled
plt.title("Age Distribution")  # Set the title of the plot
plt.xlabel("Age")  # Set the x-axis label
plt.ylabel("Count")  # Set the y-axis label
plt.show()  # Display the plot

# See how many people who used helmet survived
plt.figure(figsize=(10, 6))  # Set the figure size
sns.countplot(
    x="Survived", hue="Helmet_Used", data=df
)  # Create a count plot for survival by helmet usage
plt.title("Survival Count by Helmet")  # Set the title of the plot
plt.xlabel("Survived")  # Set the x-axis label
plt.ylabel("Count")  # Set the y-axis label
plt.legend(title="Helmet")  # Set the legend title
plt.show()  # Display the plot

# See how many people who used seatbelt survived
plt.figure(figsize=(10, 6))  # Set the figure size
sns.countplot(
    x="Survived", hue="Seatbelt_Used", data=df
)  # Create a count plot for survival by seatbelt usage
plt.title("Survival Count by Seatbelt")  # Set the title of the plot
plt.xlabel("Survived")  # Set the x-axis label
plt.ylabel("Count")  # Set the y-axis label
plt.legend(title="Seatbelt")  # Set the legend title
plt.show()  # Display the plot

# See how many people who used helmet and seatbelt survived
plt.figure(figsize=(10, 6))  # Set the figure size
sns.countplot(
    x="Survived", hue="Helmet_Used", data=df[df["Seatbelt_Used"] == "Yes"]
)  # Create a count plot for survival by helmet usage for those who used seatbelt
plt.title("Survival Count by Helmet and Seatbelt")  # Set the title of the plot
plt.xlabel("Survived")  # Set the x-axis label
plt.ylabel("Count")  # Set the y-axis label
plt.legend(title="Helmet")  # Set the legend title
plt.show()  # Display the plot

# See how speed of impact affects survival
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(
    df[df["Survived"] == 0]["Speed_of_Impact"], bins=60
)  # Create a histogram for speed of impact distribution for non-survivors
plt.title("Speed of Impact Distribution for Non-Survivors")  # Set the title of the plot
plt.xlabel("Speed of Impact")  # Set the x-axis label
plt.ylabel("Count")  # Set the y-axis label
plt.show()  # Display the plot
