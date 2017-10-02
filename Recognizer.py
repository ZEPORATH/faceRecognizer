import cv2
import threading
import numpy as np
import configData
from Queue import Queue

buffer_queue = Queue(28)


face_cascade = cv2.CascadeClassifier(configData.HAAR_FACE_CASCADE)
eye_cascade = cv2.CascadeClassifier(configData.HAAR_EYE_CASCADE)
spec_cascade = cv2.CascadeClassifier(configData.HAAR_GLASS_EYE_CASCADE)





def createCaptureDevice(source=0):
    cap = cv2.VideoCapture(source)
    return cap

def showLiveFeed(frame_buffer):
    while True:
        # ret,image = capture_device.read()
        # capture_device.stream()
        image = frame_buffer.get()
        cv2.namedWindow("LiveFeed")
        cv2.imshow("LiveFeed",image)
        if cv2.waitKey(1) & 0xFF == (ord('q')):
            break
    cv2.destroyWindow("LiveFeed")
    return

def daemonVideoCapture(capture_device,queue):
    success, image = capture_device.read()
    while success:
        success,image = capture_device.read()
        queue.put(image)

def recognize_Eigen(EIGEN):

    while True:
        frame = buffer_queue.get()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # Detect the faces and store the positions
        for (x, y, w, h) in faces:  # Frames  LOCATION X, Y  WIDTH, HEIGHT

            Face = cv2.resize((gray[y: y + h, x: x + w]),(110, 110))  # The Face is isolated and cropped

            ID, conf = EIGEN.predict(Face)  # EIGEN RECOGNITION
            print ID,conf,'Eigen'

            # ID, conf = LBPH.predict(Face)  # LBPH RECOGNITION
            # print ID,conf, 'LBPH'


def recognise_Fisher():
    pass

def recognize_LBPH():
    while True:
        frame = buffer_queue.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # Detect the faces and store the positions
        for (x, y, w, h) in faces:  # Frames  LOCATION X, Y  WIDTH, HEIGHT

            Face = cv2.resize((gray[y: y + h, x: x + w]), (110, 110))  # The Face is isolated and cropped
            ID, conf = LBPH.predict(Face)  # LBPH RECOGNITION
            print ID, 'LBPH'
def init_threads(captureDevice):
    videoThread = threading.Thread(target=daemonVideoCapture, args=(captureDevice, buffer_queue,))
    videoThread.setDaemon(True)
    videoThread.start()

    liveStreamThread = threading.Thread(target=showLiveFeed, args=(buffer_queue,))
    liveStreamThread.start()


def Main():
    import TrainerHandler
    captureDevice = createCaptureDevice(0)
    init_threads(captureDevice)
    Eigen,Lbph = TrainerHandler.generate_trainer(configData.PARENT_PATH)
    recognize_Eigen(Eigen)

if __name__=="__main__":
    Main()
