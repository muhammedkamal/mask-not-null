import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the pre-trained frontalface detection model
# Laptop:
#face_class_detector = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
# RaspberryPi:
face_class_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

size = 4                                # Scale for image resizing
model = load_model("./model-009.model") # Load the pre-trained model for mask detection, 
                                        # trained on laptop for running on laptop, and trained on RaspberryPi 
                                        # for running on RaspberryPi due to different versions of dependencies/libraries.

# Labels and colors for bound boxes
labels_dict={0:'without mask', 1:'mask'}
color_dict={0:(0,0,255), 1:(0,255,0)}

# Function for actual detection of masks logic
def detectMask(im): 
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))   # Resize image for increasing accuracy of detection
    faces = face_class_detector.detectMultiScale(mini)                  # Detecting faces in the image or video frame
    
    # Turn Off On-board "PWR" LED of RaspberryPi
    os.system('echo 0 | sudo tee /sys/class/leds/led1/brightness > /dev/null 2>&1')
    
    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        detected_face = im[y:y + h, x:x + w]
        face_to_model_size = cv2.resize(detected_face, (150, 150))
        face_norm = face_to_model_size / 255.0
        face_to_model_shape = np.reshape(face_norm, (1, 150, 150, 3))   
        face_to_model_shape = np.vstack([face_to_model_shape])          # Stack arrays in sequence vertically
        result = model.predict(face_to_model_shape)                     # Pass face after preprocessing to the face-mask classifier         
        #print(result)
                
        label = np.argmax(result, axis = 1)[0]   # Returns the indices of the maximum values along an axis.

        # if result[0][1] > 0.2:
        if label == 1:  # Mask weared, Turn off Alert LED
            os.system('echo 0 | sudo tee /sys/class/leds/led1/brightness > /dev/null 2>&1')
        else:  # No Mask weared, Turn on Alert LED
            os.system('echo 1 | sudo tee /sys/class/leds/led1/brightness > /dev/null 2>&1')
        
        # Draw bound boxes around face and put labels
        cv2.rectangle(im, (x, y), (x + w, y+h), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return np.argmax(result, axis=1)[0]
