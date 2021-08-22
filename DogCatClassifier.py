import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

target = []
images = []
flat_data = []

DATADIR = 'D:\Coding\Dog Cat Classifier\dataset\data'
CATEGORIES = ['cat', 'dog']

for category in CATEGORIES:
    class_num = CATEGORIES.index(category)  # Label encoding the values
    path = os.path.join(DATADIR, category)  # getting path of images
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

# split data into training and testing

x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=109)

param_grid = [
              {'C': [1, 10, 100, 1000], 'kernel':['linear']},
              {'C': [1, 10, 100, 1000], 'gamma':[0.001, 0.0001], 'kernel':['rbf']}
]

svc = svm.SVC(probability=True)
clf = GridSearchCV(svc, param_grid)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

#save model using Pickle library
pickle.dump(clf,open('DC_model.p', 'wb'))


# loading the saved model using pickle
model = pickle.load(open('DC_model.p', 'rb'))

# testing a brand new image
flat_data = []
url = input('Enter your URL')
img = imread(url)
img_resized = resize(img, (150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]] 
print(f'PREDICTED OUTPUT: {y_out}')