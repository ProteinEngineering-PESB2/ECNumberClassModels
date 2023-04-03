import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(df_data, columns):
    list_df = []

    for key in columns:
        data_filter = df_data[[key, "Training method"]]

        stage = ""
        performance = ""

        if "train" in key:
            stage = "Train"
            performance = key.split("train_")[1].capitalize().replace("_", "-")
        else:
            stage = "Testing"
            performance = key.split("test_")[1].capitalize().replace("_", "-")
        

        df_values = pd.DataFrame()
        df_values['Performance'] = data_filter[key]
        df_values['Training method'] = data_filter['Training method']
        df_values['Stage'] = stage
        df_values['Metric'] = performance

        list_df.append(df_values)
    
    df_concat = pd.concat(list_df, axis=0)
    return df_concat

plt.style.use("seaborn-v0_8-white")
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

df_data = pd.read_csv("../../results_training/training_results.csv")

keys = ['train_accuracy','train_recall','train_precision','train_f1_score','test_accuracy','test_recall','test_precision','test_f1_score']

df_accuracy = prepare_data(df_data, ['train_accuracy', 'test_accuracy'])
df_precision = prepare_data(df_data, ['train_precision', 'test_precision'])
df_recall = prepare_data(df_data, ['train_recall', 'test_recall'])
df_fscore = prepare_data(df_data, ['train_f1_score', 'test_f1_score'])

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14,14))

g1 =sns.boxplot(ax=axes[0][0], data=df_accuracy, x="Performance", y="Stage", hue="Training method")
axes[0][0].set_title("Accuracy")

g2 =sns.boxplot(ax=axes[0][1], data=df_precision, x="Performance", y="Stage", hue="Training method")
axes[0][1].set_title("Precision")

g3 =sns.boxplot(ax=axes[1][0], data=df_recall, x="Performance", y="Stage", hue="Training method")
axes[1][0].set_title("Recall")

g4 =sns.boxplot(ax=axes[1][1], data=df_fscore, x="Performance", y="Stage", hue="Training method")
axes[1][1].set_title("F1-score")

plt.savefig("../../save_plots/performances_by_training_method.svg")
