import pandas as pd
import numpy as np
import os, dill
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score
import tensorflow as tf

from tensorflow import keras


    
class DataCleaner:
    def __init__(self, data):
        self.data = data
        
    def clean_data_trainer(self):
        self.data.drop(['Crime ID', 'LSOA name', 'Reported by', 'Location', 'LSOA code'], 
                       axis = 1, inplace = True, errors = 'ignore')
        
        if self.data['Month'] is not None:
            self.data['Month'] = pd.to_datetime(self.data['Month'])
        
        # self.data['Month'] = pd.to_datetime(self.data['Month'])
        
        self.data.loc[:, 'Outcome type'] = self.data.loc[:, 'Outcome type'].apply(lambda x: 'prosecuted' if 'charged' in x else 'not-prosecuted')
        
        outcome_map = {'not-prosecuted':0, 'prosecuted':1}
        
        self.data['Outcome type'] = self.data['Outcome type'].map(outcome_map)
        
        self.data['year'] = self.data.loc[:, 'Month'].dt.year
        self.data['month'] = self.data.loc[:, 'Month'].dt.month
        
        self.data.drop(['Month'], axis = 1, inplace = True, errors = 'ignore')
        
        return self.data


class LoadSaveObject:
    def load_object(self, file_path):
        with open(file_path, 'rb') as f:
            return dill.load(f)
        
    def save_object(self, file_path, file_object):
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            return dill.dump(file_object, f)
        


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    
    accuracy, precision, recall, f1 = (np.round(x, 2) for x in (accuracy, precision, recall, f1))
    
    return accuracy, precision, recall, f1




class F1Score(tf.keras.metrics.Metric):

  def __init__(self, name="F1Score", **kwargs):
    super().__init__(name, **kwargs)

    # Define the precision and recall metrics.
    self.precision = tf.keras.metrics.Precision()
    self.recall = tf.keras.metrics.Recall()

  def update_state(self, y_true, y_pred, sample_weight=None):
    # Update the precision and recall metrics.
    self.precision.update_state(y_true, y_pred)
    self.recall.update_state(y_true, y_pred)

  def result(self):
    # Calculate the F1 score.
    p = self.precision.result()
    r = self.recall.result()
    return tf.math.divide(2 * p * r, p + r)

  def reset_states(self):
    self.precision.reset_state()
    self.recall.reset_state()


from tensorflow.keras import backend as K

@tf.function
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


    
def my_model():

  model = keras.Sequential([
        keras.layers.Dense(64, input_dim=59, activation='selu', 
                           kernel_initializer='he_uniform'), # initialising weights
        keras.layers.Dense(128, activation="selu"),
        keras.layers.Dense(128, activation='selu'),
        keras.layers.Dense(128, activation='selu'),
        keras.layers.Dense(96, activation='selu'),
        keras.layers.Dense(96, activation='selu'),
        keras.layers.Dense(96, activation='selu'),
        keras.layers.Dense(48, activation='selu'),
        keras.layers.Dense(48, activation='selu'),
        keras.layers.Dense(48, activation='selu'),
        keras.layers.Dense(32, activation='selu'),
        keras.layers.Dense(32, activation='selu'),
        keras.layers.Dense(24, activation='selu'),
        keras.layers.Dense(24, activation='selu'),
        keras.layers.Dense(12, activation='selu'),
        keras.layers.Dense(6, activation='selu'),
        keras.layers.Dense(4, activation='selu'),
        keras.layers.Dense(2, activation='selu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

  # The optimiser is Adam with a learning rate of 0.001:
  optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, 
                                   epsilon = 10e-8, amsgrad = True)

  # The model optimises cross entropy as its loss function and will monitor classification accuracy:
  model.compile(optimizer=optim, 
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
                metrics=f1_metric)

  # Printing model summary:
  print(model.summary())

  return model






    


