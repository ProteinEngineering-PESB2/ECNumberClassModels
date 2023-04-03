import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-white")
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

df_data = pd.read_csv("../../results_training/training_results.csv")

keys = ['train_accuracy','train_recall','train_precision','train_f1_score','test_accuracy','test_recall','test_precision','test_f1_score']

list_df = []

for key in keys:
    array_data = df_data[key]
    stage = ""
    performance = ""

    if "train" in key:
        stage = "Train"
        performance = key.split("train_")[1].capitalize().replace("_", "-")
    else:
        stage = "Testing"
        performance = key.split("test_")[1].capitalize().replace("_", "-")
    

    df_values = pd.DataFrame()
    df_values['Performance'] = array_data
    df_values['Stage'] = stage
    df_values['Metric'] = performance

    list_df.append(df_values)

df_concat = pd.concat(list_df, axis=0)
sns.boxplot(data=df_concat, x="Performance", y="Metric", hue="Stage")
plt.savefig("../../save_plots/all_performances.pdf")