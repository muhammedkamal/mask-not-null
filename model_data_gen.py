from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def model_data_gen():
    TRAINING_DIR = "./train"
    train_data_genertor = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=50,
                                    width_shift_range=0.25,
                                    height_shift_range=0.2,
                                    shear_range=0.25,
                                    zoom_range=0.25,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    train_generator = train_data_genertor.flow_from_directory(TRAINING_DIR, 
                                                        batch_size=5, 
                                                        target_size=(150, 150))
    train_generator = train_data_genertor.flow_from_directory(TRAINING_DIR, 
                                                        batch_size=5, 
                                                        target_size=(150, 150))
    VALIDATION_DIR = "./test"
    validation_data_genertor = ImageDataGenerator(rescale=1.0/255)

    validation_generator = validation_data_genertor.flow_from_directory(VALIDATION_DIR, 
                                                            batch_size=5, 
                                                            target_size=(150, 150))
    checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    return train_generator,validation_generator,checkpoint