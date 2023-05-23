import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

model = tf.keras.models.load_model('model.h5')

data = pd.read_csv('data.csv', sep=';') 

data = pd.get_dummies(data, columns=['Sex', 'Medal'])
data = data.drop(columns=['Name', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event'])

scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

X_test = data.filter(regex='Sex|Age')

predictions = model.predict(X_test)

pd.DataFrame(predictions).to_csv('predictions.csv', index=False)