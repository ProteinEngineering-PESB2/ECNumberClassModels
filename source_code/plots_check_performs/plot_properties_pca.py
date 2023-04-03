import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('bmh')
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

dict_value = {
    "Alpha structure": "Group_0_encoding_FFT.csv",
    "Beta structure": "Group_1_encoding_FFT.csv",
    "Hydropathy": "Group_2_encoding_FFT.csv",
    "Hydrophobicity": "Group_3_encoding_FFT.csv",    
    "Energy": "Group_4_encoding_FFT.csv", 
    "Secondary structure": "Group_5_encoding_FFT.csv", 
    "Volume": "Group_6_encoding_FFT.csv", 
    "Other indexes": "Group_7_encoding_FFT.csv", 
} 

path_input = "../../tsne/"

fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(20,10))

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

    if iteration != 8:
        axes[x_value][y_value].get_legend().remove()

    if iteration in [4]:
        x_value += 1
        y_value = 0
    else:
        y_value += 1
    
    iteration += 1

plt.savefig("{}pca_fft_properties.png".format(path_input), dpi=300)
