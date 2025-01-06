import pandas as pd
import matplotlib.pyplot as plt

# Set global plot style and parameters
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [20, 5]  # Set figure size
plt.rcParams["figure.dpi"] = 100  # Set figure resolution

# Load the data
df = pd.read_pickle("../data/interim/01_data_processes.pkl")  # Load processed dataset
df

# Filter dataset where 'set' equals 1
set_df = df[df["set"] == 1]

# Plot 'acc_y' for the filtered dataset
plt.plot(set_df["acc_y"])

# Plot 'acc_y' for each label in the dataset
for label, subset in df.groupby("label"):
    plt.figure()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=f"Label: {label}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
    plt.title(f"Plot for Label {label}")
    plt.xlabel("Samples")
    plt.ylabel("acc_y")
    plt.show()

# Plot first 100 samples of 'acc_y' for each label
for label, subset in df.groupby("label"):
    plt.figure()
    plt.plot(subset["acc_y"][:100].reset_index(drop=True), label=f"Label: {label}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
    plt.title(f"Plot for Label {label}")
    plt.xlabel("Samples")
    plt.ylabel("acc_y")
    plt.show()

# Filter dataset for label 'squat' and participant 'A'
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

# Plot 'acc_y' grouped by 'category'
plt.figure()
category_df.groupby("category")["acc_y"].plot()
plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
plt.title(f"Plot for Label squat (Participant A)")
plt.xlabel("Samples")
plt.ylabel("acc_y")
plt.show()

# Plot 'acc_x' grouped by 'category'
plt.figure()
category_df.groupby("category")["acc_x"].plot()
plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
plt.title(f"Plot for Label squat (Participant A)")
plt.xlabel("Samples")
plt.ylabel("acc_x")
plt.show()

# Filter dataset for label 'bench', sorted by participant
Participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

# Plot 'acc_y' grouped by participant
plt.figure()
Participant_df.groupby("participant")["acc_y"].plot()
plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
plt.title(f"Plot for Label bench")
plt.xlabel("Samples")
plt.ylabel("acc_y")
plt.show()

# Filter dataset for a specific label and participant
label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

# Plot accelerometer data for all axes
plt.figure()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot()
plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
plt.title(f"Plot for Label {label} (Participant {participant})")
plt.xlabel("Samples")
plt.ylabel("Acceleration")
plt.show()

# Get unique labels and participants
labels = df["label"].unique()
participants = df["participant"].unique()

# Plot accelerometer and gyroscope data for each label and participant
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            # Plot accelerometer data
            plt.figure()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot()
            plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
            plt.title(f"{label} ({participant}) - Accelerometer Data")
            plt.xlabel("Samples")
            plt.ylabel("Acceleration")
            plt.show()

            # Plot gyroscope data
            plt.figure()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot()
            plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
            plt.title(f"{label} ({participant}) - Gyroscope Data")
            plt.xlabel("Samples")
            plt.ylabel("Gyroscope")
            plt.show()

# Combined plots for accelerometer and gyroscope data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(
                ax=ax[0]
            )  # Accelerometer data
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])  # Gyroscope data
            ax[0].legend(
                loc="upper center", bbox_to_anchor=(0.9, 1.1), ncol=3, fancybox=True
            )
            ax[1].legend(
                loc="upper center", bbox_to_anchor=(0.9, 1.1), ncol=3, fancybox=True
            )
            ax[1].set_xlabel("Samples")
            ax[0].set_ylabel("Acceleration")
            ax[1].set_ylabel("Gyroscope")
            ax[0].set_title(f"Label={label}, Participant={participant}")
            plt.show()
