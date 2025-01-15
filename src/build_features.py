import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# Set global plot style and parameters
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [20, 10]  # Set figure size
plt.rcParams["figure.dpi"] = 100  # Set figure resolution


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../data/interim/02_outliers_removed_chauvenet.pkl")

predictor_columns = df.columns[:6].tolist()

df.info()
df.isna().sum()

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.isna().sum()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df.query("set == 10")["acc_x"].plot()

duration = df.query("set == 1").index[-1] - df.query("set == 1").index[0]
duration.total_seconds()


for set in df["set"].unique():
    duration = df.query("set == @set").index[-1] - df.query("set == @set").index[0]

    df.loc[(df["set"] == set), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()
duration_df[0] / 5
duration_df[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
Lowpass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3
df_lowpass = Lowpass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass.query("set == 45")

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="raw data")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = Lowpass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal Components")
plt.ylabel("Explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca.query("set == 45")
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared.query("set == 45")
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
