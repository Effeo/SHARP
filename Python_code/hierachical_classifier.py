import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hiclass import LocalClassifierPerNode
import scipy.io
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def demultiplex(list_to_convert):
    list_to_append = []
    for element in list_to_convert:
        if (element == np.array(["No empty", "Move", "Move in place", "Person sitting down/standing up"])).all():
            list_to_append.append(0)
        if (element == np.array(["Empty", "", "", ""])).all():
            list_to_append.append(1)
        if (element == np.array(["No empty", "Move", "Move in place", "Arm exercise"])).all():
            list_to_append.append(2)
        if (element == np.array(["No empty", "Move", "Move in place", "Jumping"])).all():
            list_to_append.append(3)
        if (element == np.array(["No empty", "No move", "Person sitting still", ""])).all():
            list_to_append.append(4)
        if (element == np.array(["No empty", "Move", "Move around", "Running"])).all():
            list_to_append.append(5)
        if (element == np.array(["No empty", "No move", "Person standing still", ""])).all():
            list_to_append.append(6)
        if (element == np.array(["No empty", "Move", "Move around", "Walking"])).all():
            list_to_append.append(7)
    list_to_append = np.array(list_to_append)
    return list_to_append

dataset = scipy.io.loadmat('./dataset/dataset.mat')
dataset = dataset['csi_buff']

raw_labels = dataset[:,-1]
dataset = np.delete(dataset, -1, 1)
labels=[]


for l in raw_labels:
    if l == 0:
        labels.append(["No empty", "Move", "Move in place", "Person sitting down/standing up"])
    elif l == 1:
        labels.append(["Empty", "", "", ""])
    elif l == 2:
        labels.append(["No empty", "Move", "Move in place", "Arm exercise"])
    elif l == 3:
        labels.append(["No empty", "Move", "Move in place", "Jumping"])
    elif l == 4:
        labels.append(["No empty", "No move", "Person sitting still", ""])
    elif l == 5:
        labels.append(["No empty", "Move", "Move around", "Running"])
    elif l == 6:
        labels.append(["No empty", "No move", "Person standing still", ""])
    elif l == 7:
        labels.append(["No empty", "Move", "Move around", "Walking"])


labels = np.array(labels, dtype=object) 

rf = RandomForestClassifier()
classifier = LocalClassifierPerNode(local_classifier=rf)
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.40, random_state = 0)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

predictions = demultiplex(predictions)
y_test = demultiplex(y_test)
cm = confusion_matrix(y_test, predictions, normalize='true') 
disp = ConfusionMatrixDisplay(np.around(cm, decimals = 2), display_labels = ['C', 'E', 'G', 'I', 'J', 'L', 'R', 'S']) 
disp.plot(cmap=plt.cm.Blues)
plt.savefig('models_outputs/hierarchical_classifier_confusion_matrix.jpeg')


