import os
import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_arc = pd.read_csv(
    "../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../data/raw/MetaMotion/*.csv")


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------


data_path = "../data/raw/MetaMotion/"
f = files[0]

f.split("/")[4].split("-")[2].strip("0123456789")

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

acc_df = acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"])
gyr_df = gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"])

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


def read_data_from_files(data_path):
    """
    Process all MetaMotion CSV files from a given directory.

    Parameters:
        data_path (str): The directory path where the CSV files are stored.

    Returns:
        tuple: A tuple containing two DataFrames, one for Accelerometer data and one for Gyroscope data.
    """

    # List all CSV files in the specified directory
    files = glob(os.path.join(data_path, "*.csv"))

    # Initialize DataFrames for Accelerometer and Gyroscope data
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    # Set counters for data grouping
    acc_set = 1
    gyr_set = 1

    # Process each file
    for f in files:
        # Extract metadata from the filename
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        # Read the file into a DataFrame
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Determine the type of data (Accelerometer or Gyroscope) and append to respective DataFrame
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df], ignore_index=True)

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df], ignore_index=True)

    # Convert epoch timestamps to datetime and set as index
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Drop unnecessary columns
    acc_df = acc_df.drop(
        columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], errors="ignore"
    )
    gyr_df = gyr_df.drop(
        columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], errors="ignore"
    )

    return acc_df, gyr_df


# Example usage:
data_path = "../../data/raw/MetaMotion/"
accelerometer_data, gyroscope_data = read_data_from_files(data_path)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([accelerometer_data.iloc[:, :3], gyroscope_data], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
data_merged.info()
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "gyr_x": "mean",
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

data_merged.resample(rule="200ms").apply(sampling)

data_resampled = (
    data_merged.groupby(pd.Grouper(freq="D"))
    .apply(lambda df: df.resample("200ms").apply(sampling).dropna())
    .reset_index(level=0, drop=True)
)
data_resampled.info()

data_resampled["set'"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Numeric Summary
# --------------------------------------------------------------

import numpy as np


def summarize_numeric_data(df):
    """
    Generate a summary for numeric columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A summary DataFrame containing statistics for numeric columns.
    """
    # Select only numeric columns
    numeric_data = df.select_dtypes(include=[np.number])

    # Function to compute summary statistics for a single column
    def my_numeric_summary(x):
        return pd.Series(
            {
                "n": len(x),
                "unique": x.nunique(),
                "missing": x.isna().sum(),
                "mean": x.mean(),
                "min": x.min(),
                "Q1": np.percentile(x.dropna(), 25),
                "median": x.median(),
                "Q3": np.percentile(x.dropna(), 75),
                "max": x.max(),
                "sd": x.std(),
            }
        )

    # Compute the summary for numeric columns
    summary = (
        numeric_data.apply(my_numeric_summary)
        .T.reset_index()  # Transpose for better readability
        .rename(columns={"index": "variable"})
    )

    # Add percentage columns for missing and unique values
    summary = summary.assign(
        missing_pct=(summary["missing"] / summary["n"]) * 100,
        unique_pct=(summary["unique"] / summary["n"]) * 100,
    )

    # Reorganize columns for clarity
    summary = summary[
        [
            "variable",
            "n",
            "missing",
            "missing_pct",
            "unique",
            "unique_pct",
            "mean",
            "min",
            "Q1",
            "median",
            "Q3",
            "max",
            "sd",
        ]
    ]

    return summary


summarize_numeric_data(data_resampled)
data_resampled.describe()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processes.pkl")
