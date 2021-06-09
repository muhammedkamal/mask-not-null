from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout

def model_constructtion():
    model = Sequential([
        Conv2D(100, (3,3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(2,2), 
        Conv2D(100, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model