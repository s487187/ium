import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


model_path = 'model.h5'
test_data_path = 'data.csv'
metrics_file_path = 'metrics.txt'
plot_path = 'plot.png'

model = tf.keras.models.load_model(model_path)

test_data = pd.read_csv(test_data_path, sep=';')
test_data = pd.get_dummies(test_data, columns=['Sex', 'Medal'])
test_data = test_data.drop(columns=['Name', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event'])

scaler = MinMaxScaler()
test_data = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)

X_test = test_data.filter(regex='Sex|Age')
y_test = test_data.filter(regex='Medal')
y_test = pd.get_dummies(y_test)

X_test = X_test.fillna(0)
y_test = y_test.fillna(0)

y_pred = model.predict(X_test)

top_1_accuracy = tf.keras.metrics.categorical_accuracy(y_test, y_pred)
top_5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=5)

if os.path.exists(metrics_file_path):
    metrics_df = pd.read_csv(metrics_file_path)
else:
    metrics_df = pd.DataFrame(columns=['top_1_accuracy', 'top_5_accuracy'])

new_row = pd.DataFrame([{'top_1_accuracy': np.mean(top_1_accuracy.numpy()), 'top_5_accuracy': np.mean(top_5_accuracy.numpy())}])
metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
metrics_df.to_csv(metrics_file_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(metrics_df['top_1_accuracy'], label='Top-1 Accuracy')
plt.plot(metrics_df['top_5_accuracy'], label='Top-5 Accuracy')
plt.legend()
plt.savefig(plot_path)