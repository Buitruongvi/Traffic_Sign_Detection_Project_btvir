# Traffic Sign Detection Project

Traffic Sign Detection is an application that involves using algorithms related to Object Detection to detect traffic signs on the road. 
Traffic Sign Detection models are commonly used in various applications such as self-driving cars, Advanced Driver Assistance Systems (ADAS), and more. 
A typical Traffic Sign Detection program consists of two stages: locating the position of the traffic sign and recognizing the name or content of the sign. 
Therefore, a high-precision program needs to excel in both of these components.

![image](https://github.com/Buitruongvi/Traffic_Sign_Detection_Project_btvir/assets/49474873/6ccfbad8-e01c-41c5-bddf-09e0d9bda536)

In this project, we will build a Traffic Sign Detection program using a Support Vector Machine (SVM) model. The input and output of the program are as follows:
- Input: An image containing traffic signs. This image can be provided in various image file formats, such as JPEG, PNG, or any other image format.
- Output: The program will provide information about the coordinates and names (classes) of all the traffic signs present in the image.

## 1. Download the dataset: 
Please download the dataset for the Traffic Sign Detection problem from [here](https://drive.google.com/file/d/1YJiHQeLotsaXAXCtLLKBHPaawqKiSC5b/view).
This dataset contains 877 images categorized into 4 classes: 'trafficlight', 'stop', 'speedlimit', and 'crosswalk'.

## 2. Import the necessary libraries:
```python
import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from skimage.transform import resize
from skimage import feature
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn. model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### 3. Loading Data:
We proceed to read the image files and labels into two separate lists, corresponding to the X and y pairs in our traffic sign classification problem. After extracting the contents of the .zip file, we have two folders with the following contents:
- Images: The folder containing image files.
- Annotations: The folder containing `.xml` files, which are label files containing information about the coordinates and classes of objects within the images, corresponding to the filenames in the images folder.
```python
annotations_dir = 'annotations'
img_dir = 'images'

img_lst = []
label_lst = []

for xml_file in os.listdir(annotations_dir):
  xml_filepath = os.path.join(annotations_dir, xml_file)

  tree = ET.parse(xml_filepath)
  root = tree.getroot()

  folder = root.find('folder').text
  img_filename = root.find('filename').text
  img_filepath = os.path.join(img_dir, img_filename )
  img = cv2.imread(img_filepath)

  for obj in root.findall('object'):
    classname = obj.find('name').text
    if classname == 'trafficlight':
      continue

    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)

    object_img = img[ymin:ymax, xmin:xmax]
    img_lst.append(object_img)
    label_lst.append(classname)

print('Number of object: ', len(img_lst))
print('Class names: ', list(set(label_lst)))
```

## 4. Image Preprocessing
To enhance the accuracy of the SVM model, we will construct a preprocessing function for the input images to create a better representation (feature) for the images. Specifically, we will utilize the Histogram of Oriented Gradients (HOG) feature in this context.
```python
def preprocess_img(img):
  if len(img.shape) > 2:
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

  img = img.astype(np.float32)

  resized_img = resize(
      img,
      output_shape = (32, 32),
      anti_aliasing=True
  )

  hog_feature = feature.hog(
      resized_img,
      orientations=9,
      pixels_per_cell=(8, 8),
      cells_per_block=(2, 2),
      transform_sqrt=True,
      block_norm="L2",
      feature_vector=True
  )

  return hog_feature
```
![image](https://github.com/Buitruongvi/Traffic_Sign_Detection_Project_btvir/assets/49474873/0a827d8e-451c-4c06-a1cd-f71666f2069e)

Image Preprocessing: With the `preprocess_img()` function, we perform image preprocessing on the entire input images as follows:
```python
img_features_lst = []
for img in img_lst:
  hog_feature = preprocess_img(img)
  img_features_lst.append(hog_feature)

img_features = np.array( img_features_lst )
```

## 5. Encode Label: 
Currently, the labels are in string format, and we need to convert them into numerical format to align with the model training requirements. Here, we use `LabelEncoder()` to map class names to corresponding numbers, such as 0, 1, 2, and so on:
```python
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(label_lst)
```
## 6. Train the Model: 
After completing all the necessary steps, we proceed to train the SVM model on the training dataset:
```python
random_state = 0
test_size = 0.3
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    img_features,
    encoded_labels,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

clf = SVC(
    kernel='rbf',
    random_state=random_state,
    probability=True,
    C=0.5
)

clf.fit(X_train , y_train)
```
## 7. Evaluate the Model: 
We assess the trained model on the validation dataset:
```python
y_pred = clf.predict(X_val)
score = accuracy_score(y_pred, y_val)
print('Evaluation results on val set')
print('Accuracy: ', score)
```
```
Evaluation results on val set
Accuracy:  0.9659442724458205
```
![image](https://github.com/Buitruongvi/Traffic_Sign_Detection_Project_btvir/assets/49474873/ce6ad3ea-a3a5-46a5-924e-749cac8976c0)

# References


























