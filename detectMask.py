import cv2
import numpy as np
from tensorflow.keras.models import load_model
face_class_detector = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
size = 4
model=load_model("./model-009.model")

def detectMask(im):
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    faces = face_class_detector.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        detected_face = im[y:y + h, x:x + w]
        face_to_model_size = cv2.resize(detected_face, (150, 150))
        face_norm = face_to_model_size / 255.0
        face_to_model_shape = np.reshape(face_norm, (1, 150, 150, 3))
        face_to_model_shape = np.vstack([face_to_model_shape])
        result = model.predict(face_to_model_shape)
        return np.argmax(result, axis=1)[0]
