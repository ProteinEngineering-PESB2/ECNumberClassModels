import pandas as pd

df_results = pd.read_csv("../../results_training/training_results.csv")

df_results['overfitting_accuracy'] = 1-(df_results['train_accuracy']*df_results['test_accuracy'])
df_results['overfitting_precision'] = 1-(df_results['train_precision']*df_results['test_precision'])
df_results['overfitting_recall'] = 1-(df_results['train_recall']*df_results['test_recall'])
df_results['overfitting_f1_score'] = 1-(df_results['train_f1_score']*df_results['test_f1_score'])

df_results.to_csv("../../results_training/training_results_with_over_rate.csv", index=False)
