import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

df = pd.read_pickle("../data/interim/01_data_processes.pkl")
df

set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
