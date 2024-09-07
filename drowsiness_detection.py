from google.colab import drive
drive.mount('/content/drive')
!pip install mediapipe
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import matplotlib.pyplot as plt
import mediapipe as mp
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates
%matplotlib inline
os.makedirs('./Fatigue Subjects')
os.makedirs('./Active Subjects')
# Image preprocessing :
### our preprocessing will include
- Detecting faces from images
- Drawing landmarks on our images to increase performance
- Resizing our images
- LabelEncoding
- Image Augmantation
# Landmarks :
We will use mediapipe to draw landmarks on our images after detecting faces and 
croping them
# Landmark points corresponding to left eye
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
# flatten and remove duplicates
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))
# Landmark points corresponding to right eye
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
# Combined for plotting - Landmark points for both eye
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
# The chosen 12 points: P1, P2, P3, P4, P5, P6
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
IMG_SIZE=145
i=0
def draw(
 *,n=i,
 img_dt,cat,
 img_eye_lmks=None,
 img_eye_lmks_chosen=None,
 face_landmarks=None,
 ts_thickness=1,
 ts_circle_radius=2,
 lmk_circle_radius=3,
 name="1",
):
 # For plotting Face Tessellation
 image_drawing_tool = img_dt
 # For plotting all eye landmarks
 image_eye_lmks = img_dt.copy() if img_eye_lmks is None else img_eye_lmks
 # For plotting chosen eye landmarks
 img_eye_lmks_chosen = img_dt.copy() if img_eye_lmks_chosen is None else 
img_eye_lmks_chosen
 # Initializing drawing utilities for plotting face mesh tessellation
 connections_drawing_spec = mp_drawing.DrawingSpec(
 thickness=ts_thickness,
 circle_radius=ts_circle_radius,
 color=(255, 255, 255)
 )
 # Draw landmarks on face using the drawing utilities.
 mp_drawing.draw_landmarks(
 image=image_drawing_tool,
 landmark_list=face_landmarks,
 connections=mp_facemesh.FACEMESH_TESSELATION,
 landmark_drawing_spec=None,
 connection_drawing_spec=connections_drawing_spec,
 )
 # Get the object which holds the x, y, and z coordinates for each landmark
 landmarks = face_landmarks.landmark
 # Iterate over all landmarks.
 # If the landmark_idx is present in either all_idxs or all_chosen_idxs,
 # get the denormalized coordinates and plot circles at those coordinates.
 for landmark_idx, landmark in enumerate(landmarks):
 if landmark_idx in all_idxs:
 pred_cord = denormalize_coordinates(landmark.x,
 landmark.y,
imgW, imgH)
 cv2.circle(image_eye_lmks,
 pred_cord,
 lmk_circle_radius,
 (255, 255, 255),
 -1
 )
 if landmark_idx in all_chosen_idxs:
 pred_cord = denormalize_coordinates(landmark.x,
 landmark.y,
imgW, imgH)
 cv2.circle(img_eye_lmks_chosen,
 pred_cord,
 lmk_circle_radius,
 (255, 255, 255),
 -1
 )
 if cat=='Fatigue Subjects':
 cv2.imwrite(str('./Fatigue Subjects/'+str(n)+'.jpg'), image_drawing_tool)
 else:
 cv2.imwrite(str('./Active Subjects/'+str(n)+'.jpg'), image_drawing_tool)
 resized_array = cv2.resize(image_drawing_tool, (IMG_SIZE, IMG_SIZE))
 return resized_array
imgH, imgW, _=0,0,0
def landmarks(image,category,i):
 resized_array=[]
 IMG_SIZE = 145
 image = np.ascontiguousarray(image)
 imgH, imgW, _ = image.shape
 # Running inference using static_image_mode
 with mp_facemesh.FaceMesh(
 static_image_mode=True, # Default=False
 max_num_faces=1, # Default=1
 refine_landmarks=False, # Default=False
 min_detection_confidence=0.5, # Default=0.5
 min_tracking_confidence= 0.5,) as face_mesh:
 results = face_mesh.process(image)
 # If detections are available.
 if results.multi_face_landmarks:
 for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
 resized_array= draw(img_dt=image.copy(), cat=category, 
n=i,face_landmarks=face_landmarks)
 return resized_array
def face_for_yawn(direc="/content/drive/MyDrive/input/drowsiness-predictiondataset/0 FaceImages",
 face_cas_path="/content/drive/MyDrive/input/predictionimages/haarcascade_frontalface_default.xml",
 batch_size=10):
 yaw_no=[]
 i=1
 IMG_SIZE = 145
 categories = ["Fatigue Subjects", "Active Subjects"]
 for category in categories:
 path_link = os.path.join(direc, category)
 class_num1 = categories.index(category)
 print(class_num1)
 for image in os.listdir(path_link):
 image_array = cv2.imread(os.path.join(path_link, image), 
cv2.IMREAD_COLOR)
 face_cascade = cv2.CascadeClassifier(face_cas_path)
 faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
 for (x, y, w, h) in faces:
 img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
 roi_color = img[y:y+h, x:x+w]
 land_face_array=landmarks(roi_color,category,i)
 yaw_no.append([land_face_array, class_num1])
 i=i+1
 return yaw_no
yawn_no_yawn = face_for_yawn()
dir_path = r'/content/drive/MyDrive/input/drowsiness-prediction-dataset/0 
FaceImages/Active Subjects'
print("Number of Active images :")
print(len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, 
entry))]))
dir_path = r'/content/drive/MyDrive/input/drowsiness-prediction-dataset/0 
FaceImages/Fatigue Subjects'
print("Number of Fatigue images :")
print(len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, 
entry))]))
### Our images will be like this:
categories = ["Fatigue Subjects", "Active Subjects"]
for category in categories:
 for idx, img in enumerate(os.listdir(f'./{category}')):
 if idx > 5:
 break
 img_file = cv2.imread(f'./{category}/{img}')
 plt.imshow(img_file)
 plt.show()
 plt.close()
import os
import cv2
import matplotlib.pyplot as plt
categories = ["Fatigue Subjects", "Active Subjects"]
for category in categories:
 print(f"Displaying images for category: {category}")
 for idx, img_name in enumerate(os.listdir(category)):
 if idx > 5:
 break
 img_path = os.path.join(category, img_name)
 if not os.path.isfile(img_path):
 print(f"File not found: {img_path}")
 continue
 img_file = cv2.imread(img_path)
 if img_file is None:
 print(f"Failed to load image: {img_path}")
 continue
 plt.imshow(cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB))
 plt.title(f"{category} - {idx}")
 plt.show()
 plt.close()
### Resizing images
import os
import time
def face_for_yawn(direc="./"):
 yaw_no=[]
 i=1
 IMG_SIZE = 145
 categories = ["Fatigue Subjects", "Active Subjects"]
 for category in categories:
 path_link = os.path.join(direc, category)
 class_num1 = categories.index(category)
 print(class_num1)
 for image in os.listdir(path_link):
 image_array = cv2.imread(os.path.join(path_link, image), 
cv2.IMREAD_COLOR)
 resized_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
 yaw_no.append([resized_array, class_num1])
 #print('image face number '+str(i))
 #i=i+1
 return yaw_no
yawn_no_yawn = face_for_yawn()
## separate label and features
X = []
y = []
for feature, label in yawn_no_yawn:
 X.append(feature)
 y.append(label)
## Reshape the array
X = np.array(X)
X = X.reshape(-1, 145, 145, 3)
## LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_bin = LabelEncoder()
y = label_bin.fit_transform(y)
y = np.array(y)
# Splitting
from sklearn.model_selection import train_test_split
seed = 42
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, 
test_size=test_size)
len(X_test)
len(X_train)
### import some dependencies
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, 
MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# Data Augmentation
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, 
horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)
train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)
# YOLO Model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, 
MaxPooling2D, Flatten, Dense, Reshape
def YOLO_model(input_shape):
 inputs = tf.keras.Input(shape=input_shape)
 # Convolutional Backbone
 x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
 x = BatchNormalization()(x)
 x = MaxPooling2D()(x)
 x = Conv2D(32, 5, activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = MaxPooling2D()(x)
 x = Conv2D(64, 10, activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = MaxPooling2D()(x)
 x = Conv2D(128, 12, activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = MaxPooling2D()(x)
 # Additional layers for bounding box regression and classification
 x = Conv2D(256, 3, activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = MaxPooling2D()(x)
 x = Conv2D(512, 3, activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = MaxPooling2D()(x)
 x = Conv2D(1024, 3, activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 # Flatten and dense layers
 x = Flatten()(x)
 x = Dense(512, activation='relu')(x)
 x = Dropout(0.5)(x)
 x = Dense(256, activation='relu')(x)
 # Output layer
 predictions = Dense(1, activation='sigmoid')(x)
 model = tf.keras.Model(inputs, predictions)
 return model
# Example usage:
input_shape = (145, 145, 3) # Example input shape (adjust based on your data)
model = YOLO_model(input_shape)
model.summary()
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()
plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()
# You can evaluate or predict on a dataset.
print("Evaluate")
result = model.evaluate(test_generator)
dict(zip(model.metrics_names, result))
model.save('my_model.h5')
#model = tf.keras.models.load_model('my_model.h5')
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
binary1 = np.array([[3427,252],[44,2331]])
fig, ax = plot_confusion_matrix(conf_mat=binary1,show_absolute=True,
 show_normed=True,
 colorbar=True)
plt.show()
# Visualizing our CNN architecture
!pip install visualkeras
import visualkeras
visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model, to_file='output.png') # write to disk
visualkeras.layered_view(model, to_file='output.png').show() # write and show
visualkeras.layered_view(model,legend=True)
model.summary()
pip install keyboard
!pip install food_facts
!pip install pygobject
!pip install cv
!pip install cvlib
!pip install --upgrade pip setuptools
!pip install pygobject
!pip install cvlib
!pip install gtts
!pip install playsound
from google.colab import drive
drive.mount('/content/drive')
!pip install mediapipe
import os
import cv2
import numpy as np
import mediapipe as mp
# Load MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
# Function to preprocess image for drowsiness detection
def preprocess_image(image):
 # Resize image for consistency (e.g., to 640x480)
 resized_image = cv2.resize(image, (640, 480))
 # Convert to RGB (MediaPipe requires RGB format)
 rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
 return rgb_image
# Function to calculate eye aspect ratio (EAR)
def calculate_eye_aspect_ratio(eye_landmarks):
 # Extract coordinates from eye landmarks
 landmarks_array = np.array(eye_landmarks)
 # Compute Euclidean distances between specific landmark points
 A = np.linalg.norm(landmarks_array[1] - landmarks_array[5]) # Vertical distance 
(top to bottom of eye)
 B = np.linalg.norm(landmarks_array[2] - landmarks_array[4]) # Vertical distance 
(top to bottom of eye)
 C = np.linalg.norm(landmarks_array[0] - landmarks_array[3]) # Horizontal 
distance (left to right of eye)
 # Calculate eye aspect ratio (EAR)
 ear = (A + B) / (2 * C)
 return ear
def detect_drowsiness(image):
 # Process image with MediaPipe Face Mesh
 results = face_mesh.process(image)
 if results.multi_face_landmarks:
 for face_landmarks in results.multi_face_landmarks:
 # Extract eye landmarks
 left_eye_landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * 
image.shape[0])) for landmark in face_landmarks.landmark[263:278]]
 right_eye_landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * 
image.shape[0])) for landmark in face_landmarks.landmark[362:377]]
 # Calculate eye aspect ratio (EAR) for both eyes
 left_ear = calculate_eye_aspect_ratio(left_eye_landmarks)
 right_ear = calculate_eye_aspect_ratio(right_eye_landmarks)
 # Print EAR values for debugging
 print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}")
 # Adjust EAR threshold based on observations
 if left_ear < 2.3 and right_ear < 0.7: # Experiment with different thresholds
 cv2.putText(image, "Drowsy", (10, 30), 
cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
 else:
 cv2.putText(image, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
0.8, (0, 255, 0), 2)
 return image
# Directory containing images
image_dir = '/content'
# Iterate over files in the directory
for filename in os.listdir(image_dir):
 file_path = os.path.join(image_dir, filename)
 # Check if the file is an image
 if not os.path.isfile(file_path) or not file_path.endswith(('.jpg', '.jpeg', '.png')):
 continue
 # Load image
 frame = cv2.imread(file_path)
 # Check if the image was loaded successfully
 if frame is None:
 print(f"Error: Unable to load image {filename}")
 continue
 # Preprocess image for drowsiness detection
 preprocessed_frame = preprocess_image(frame.copy())
 # Detect drowsiness based on eye aspect ratio (EAR) using MediaPipe Face Mesh
 frame_with_drowsiness = detect_drowsiness(preprocessed_frame)
 # Display the image with drowsiness assessment using cv2_imshow() in Colab
 cv2_imshow(frame_with_drowsiness