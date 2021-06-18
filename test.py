import cv2
import detectMask

# Laptop webcam
# webcam = cv2.VideoCapture(0) #Use camera 0 TODO : use another camera by usb

# Mobile camera live stream
iPhoneCamera = cv2.VideoCapture('http://192.168.1.7:8080/video')

def main():
    while True:
        # (rval, im) = webcam.read()
        (rval, im) = iPhoneCamera.read()
        im = cv2.flip(im,1,1)               # Flip to act as a mirror
        print(detectMask.detectMask(im))
        
        cv2.imshow('LIVE', im)
        key = cv2.waitKey(10)
        # if Esc key is press then break out of the loop 
        if key == 27: #The Esc key #TODO you can add a push button instead to exit the prog
            break
    # webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
