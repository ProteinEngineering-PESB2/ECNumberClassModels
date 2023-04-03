import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('bmh')
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

dict_value = {
    "Bepler": "bepler.csv",
    "Esm1b": "esm1b.csv",
    "Fastext": "fasttext.csv",
    "Glove": "glove.csv",    
    "Plus rnn": "plus_rnn.csv", 
    "Prottrans-Bert": "prottrans_bert_bfd.csv", 
} 

path_input = "../../tsne/"

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(18,10))

x_value = 0
y_value = 0
iteration = 1

for element in dict_value:
    print("Processing element: ", element)
    df_data = pd.read_csv(path_input+dict_value[element])

    df_data = df_data.rename(columns={"response" : "EC-class"})
    g = sns.scatterplot(ax=axes[x_value][y_value], x = "p1", y = "p2", data=df_data, hue="EC-class")

    axes[x_value][y_value].set_title(element)

    axes[x_value][y_value].set_ylabel(' ')
    axes[x_value][y_value].set_xlabel(' ')

    if iteration != 6:
        axes[x_value][y_value].get_legend().remove()

    if iteration in [3]:
        x_value += 1
        y_value = 0
    else:
        y_value += 1
    
    iteration += 1

plt.savefig("{}summary_plots_data.png".format(path_input), dpi=300)
