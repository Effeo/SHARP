import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import time
import pickle
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf

dataset = scipy.io.loadmat('./dataset/dataset.mat')
dataset = dataset['csi_buff']

labels = dataset[:,-1]
dataset = np.delete(dataset, -1, 1)
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.40, random_state = 0)
name = 'dl_model'

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(8)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model = RandomForestClassifier(random_state = 12)
start = time.process_time()
model.fit(x_train, y_train)
end = time.process_time() - start
print(end)
predictions = model.predict(x_test)
cm = confusion_matrix(y_test, predictions, normalize='true')
disp = ConfusionMatrixDisplay(np.around(cm, decimals = 2), display_labels=['C', 'E', 'G', 'I', 'J', 'L', 'R', 'S'])
disp.plot(cmap=plt.cm.Blues)
plt.savefig('models_outputs/confusion_matrix_' + name + '.jpeg')

pickle.dump(model, open('models_outputs/' + name + '.pkl', 'wb'))


