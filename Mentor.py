import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = "../Data_Academy/accident.csv"
df = pd.read_csv(data_path)

df.head()


# Data Visualization
# See whether the gender of the driver has an impact on the survival
plt.figure(figsize=(10, 6))
sns.countplot(x="Survived", hue="Gender", data=df)
plt.title("Survival Count by Gender")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Gender")
plt.show()


# See whether the age of the driver has an impact on the survival
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# See how many people who used helmet survived
plt.figure(figsize=(10, 6))
sns.countplot(x="Survived", hue="Helmet_Used", data=df)
plt.title("Survival Count by Helmet")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Helmet")
plt.show()


# See how many people who used seatbelt survived
plt.figure(figsize=(10, 6))
sns.countplot(x="Survived", hue="Seatbelt_Used", data=df)
plt.title("Survival Count by Seatbelt")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Seatbelt")
plt.show()

# See how many people who used helmet and seatbelt survived
plt.figure(figsize=(10, 6))
sns.countplot(x="Survived", hue="Helmet_Used", data=df[df["Seatbelt_Used"] == "Yes"])
plt.title("Survival Count by Helmet and Seatbelt")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Helmet")
plt.show()

# see how speed of impact affects survival
plt.figure(figsize=(10, 6))
sns.histplot(df[df["Survived"] == 0]["Speed_of_Impact"], bins=60)
plt.title("Speed of Impact Distribution for Non-Survivors")
plt.xlabel("Speed of Impact")
plt.ylabel("Count")
plt.show()
