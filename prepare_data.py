import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_path = 'data.csv'
processed_data_path = 'processed_data.csv'

data = pd.read_csv(data_path, sep=';')

data = pd.get_dummies(data, columns=['Sex', 'Medal'])

data = data.drop(columns=['Name', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event'])

scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

data.to_csv(processed_data_path, index=False)
