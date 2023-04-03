import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-white")
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

df_data = pd.read_csv("../../results_training/training_results_with_over_rate.csv")

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20,14))
g1 =sns.boxplot(ax=axes[0][0], data=df_data, x="Overfitting-Accuracy", y="Method")
axes[0][0].set_title("Accuracy")

g1 =sns.boxplot(ax=axes[1][0], data=df_data, x="Overfitting-Precision", y="Method")
axes[0][1].set_title("Precision")

g1 =sns.boxplot(ax=axes[0][1], data=df_data, x="Overfitting-Recall", y="Method")
axes[1][0].set_title("Recall")

g1 =sns.boxplot(ax=axes[1][1], data=df_data, x="Overfitting-F1 score", y="Method")
axes[1][1].set_title("F1-Score")

plt.savefig("../../save_plots/overffiting_by_method.svg")