import json

with open("../../performances_cnn/CNN_B_esm1b_scale_0.json") as cnnb_results:
    dict_cnn_b_results = json.load(cnnb_results)

with open("../../performances_cnn/CNN_A_esm1b_scale_0.json") as cnna_results:
    dict_cnn_a_results = json.load(cnna_results)

with open("../../performances_cnn/CNN_C_esm1b_scale_0.json") as cnnc_results:
    dict_cnn_c_results = json.load(cnnc_results)


accuracy_b = dict_cnn_b_results['accuracy_history']
loss_b = dict_cnn_b_results['loss_history']

accuracy_a = dict_cnn_a_results['accuracy_history']
loss_a = dict_cnn_a_results['loss_history']

accuracy_c = dict_cnn_c_results['accuracy_history']
loss_c = dict_cnn_c_results['loss_history']

x_value = [i for i in range(1, len(accuracy_a)+1)]

import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use("bmh")
plt.rc('axes', grid=False, facecolor="white")
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(21, 7))

plt.subplot(1, 3, 1)
plt.plot(x_value, accuracy_b, label='Accuracy')
plt.plot(x_value, loss_b, label='Loss Accuracy')
plt.legend(loc='lower right')
plt.title('Training performance for Architecture B')

plt.subplot(1, 3, 2)
plt.plot(x_value, accuracy_a, label='Accuracy')
plt.plot(x_value, loss_a, label='Loss Accuracy')
plt.legend(loc='lower right')
plt.title('Training performance for Architecture A')

plt.subplot(1, 3, 3)
plt.plot(x_value, accuracy_c, label='Accuracy')
plt.plot(x_value, loss_c, label='Loss Accuracy')
plt.legend(loc='lower right')
plt.title('Training performance for Architecture C')

plt.savefig("../../save_plots/history_train.svg")