# Simulating a Smart Door using Face Mask Detection

## Project description
In this project we implemented facial and face mask detection algorithms that can be utilized in a smart door system where a door of a shop or a mosque for example opens only when wearing the mask. 
Our approach was to use OpenCV and Deep Learning to train a neural network model on a dataset of images, then test the model using new images that are not in the dataset, and to support real-time video from a wireless mobile phone stream which simulates the case in an automatic door system. 
We deployed the model on a RaspberryPi board (embedded microprocessor), did the training and detection processes on-board and exported received camera stream to a monitor after processing it. 
Simulating the opening and closing of the door is achieved by turning on an on-board LED simulating a signal that is controlled by an automatic door system microcontroller. 
Using a camera instead of a passive infrared sensor in automatic door systems can be used for security surveillance as well.

### By MASK!=NULL Team
