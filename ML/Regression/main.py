import pandas as pd
from LinearRegression import LinearRegression
import numpy as np
import matplotlib as plt

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)


# Load the data
path = "../Regression/regts_data/tunnel.csv"
df = pd.read_csv(path, parse_dates=["Day"])

# Set index
df = df.set_index("Day")


df = df.to_period()
df["Time"] = np.arange(len(df.index))


# Training data
X = df.loc[:, ["Time"]]  # features
y = df.loc[:, "NumVehicles"]  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)

# Plotting the fitted values over time
ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title("Time Plot of Tunnel Traffic")


# Lag feature
df["Lag_1"] = df["NumVehicles"].shift(1)


X = df.loc[:, ["Lag_1"]]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, "NumVehicles"]  # create the target
y, X = y.align(X, join="inner")  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax.plot(X["Lag_1"], y, ".", color="0.25")
ax.plot(X["Lag_1"], y_pred)
ax.set_aspect("equal")
ax.set_ylabel("NumVehicles")
ax.set_xlabel("Lag_1")
ax.set_title("Lag Plot of Tunnel Traffic")


ax = y.plot(**plot_params)
ax = y_pred.plot()
