import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [20, 5]
plt.rcParams["figure.dpi"] = 100


df = pd.read_pickle("../data/interim/01_data_processes.pkl")
df

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])

for label, subset in df.groupby("label"):
    plt.figure()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=f"Label: {label}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title(f"Plot for Label {label}")
    plt.xlabel("Samples")
    plt.ylabel("acc_y")
    plt.show()

for label, subset in df.groupby("label"):
    plt.figure()
    plt.plot(subset["acc_y"][:100].reset_index(drop=True), label=f"Label: {label}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title(f"Plot for Label {label}")
    plt.xlabel("Samples")
    plt.ylabel("acc_y")
    plt.show()


category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

plt.figure()
category_df.groupby("category")["acc_y"].plot()
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.title(f"Plot for Label {label}")
plt.xlabel("Samples")
plt.ylabel("acc_y")
plt.show()


plt.figure()
category_df.groupby("category")["acc_x"].plot()
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.title(f"Plot for Label {label}")
plt.xlabel("Samples")
plt.ylabel("acc_y")
plt.show()


Participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

plt.figure()
Participant_df.groupby("participant")["acc_y"].plot()
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.title(f"Plot for Label {label}")
plt.xlabel("Samples")
plt.ylabel("acc_y")
plt.show()


label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)
