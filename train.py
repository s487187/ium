import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from mlflow.models import Model
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import os
import tensorflow as tf
from tensorflow.python.framework import tensor_spec
import numpy as np

os.environ["SACRED_NO_GIT"] = "1"

ex = Experiment('s487187-training', interactive=True, save_git_info=False)
ex.observers.append(MongoObserver(url='mongodb://admin:IUM_2021@172.17.0.1:27017', db_name='sacred'))

mlflow.set_tracking_uri("http://172.17.0.1:5000")
mlflow.set_experiment("s487187")


@ex.config
def my_config():
    data_file = 'data.csv' 
    model_file = 'model.h5'
    epochs = 10
    batch_size = 32
    test_size = 0.2
    random_state = 42

@ex.capture
def train_model(data_file, model_file, epochs, batch_size, test_size, random_state):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    data = pd.read_csv(data_file, sep=';', header=0)

    print('Total rows:', len(data))
    print('Rows with medal:', len(data.dropna(subset=['Medal'])))

    data = pd.get_dummies(data, columns=['Sex', 'Medal'])
    data = data.drop(columns=['Name', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event'])

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    X = data.filter(regex='Sex|Age')
    y = data.filter(regex='Medal')
    y = pd.get_dummies(y)

    X = X.fillna(0)
    y = y.fillna(0)

    y = y.values

    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=random_state)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu')) 
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax')) 

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test accuracy:', accuracy)
    print('Test loss:', loss)

    model.save("model.h5")

    X_train_numpy = X_train.values
    signature = infer_signature(X_train_numpy, model.predict(X_train_numpy))
    input_example = X_train.head(1).values

    # input_signature = {
    #     'input': tensor_spec.TensorSpec(shape=X_train.iloc[0].shape, dtype=X_train.dtypes[0])
    # }

    mlflow.keras.log_model(model, "model")
    mlflow.log_artifact("model.h5")

    # Use the ndarray form for infer_signature and input_example
    signature = infer_signature(X_train_numpy, model.predict(X_train_numpy))
    input_example = X_train.head(1).values
    mlflow.keras.save_model(model, "model", signature=signature, input_example=input_example)

    return accuracy

@ex.main
def run_experiment():
    accuracy = train_model()
    ex.log_scalar('accuracy', accuracy)  
    ex.add_artifact('model.h5')  

ex.run()
