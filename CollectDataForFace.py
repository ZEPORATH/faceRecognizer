import cv2
import logging
import numpy as np
import threading
import math
import time
from Queue import Queue
import configData

buffer_queue = Queue(28)  # Create a buffer Queue of size 128 frames


#Creates directory structure for new user.
def create_new_Record(user_ID):
    import os
    recordPath = configData.PARENT_PATH+user_ID

    if not os.path.exists(recordPath):
        try:
            os.makedirs(recordPath)
            return os.path.dirname(recordPath)
        except OSError as exc:
                raise exc
        except Exception as e:
            raise e
    else:
        return configData.FOLDER_EXISTS

def DetectEyes(Image):
    glass_cas = cv2.CascadeClassifier(configData.HAAR_GLASS_EYE_CASCADE)
    face = cv2.CascadeClassifier(configData.HAAR_FACE_CASCADE)
    Theta = 0
    rows, cols = Image.shape
    glass = glass_cas.detectMultiScale(Image)                                               # This ditects the eyes
    for (sx, sy, sw, sh) in glass:
        if glass.shape[0] == 2:                                                             # The Image should have 2 eyes
            if glass[1][0] > glass[0][0]:
                DY = ((glass[1][1] + glass[1][3] / 2) - (glass[0][1] + glass[0][3] / 2))    # Height diffrence between the glass
                DX = ((glass[1][0] + glass[1][2] / 2) - glass[0][0] + (glass[0][2] / 2))    # Width diffrance between the glass
            else:
                DY = (-(glass[1][1] + glass[1][3] / 2) + (glass[0][1] + glass[0][3] / 2))   # Height diffrence between the glass
                DX = (-(glass[1][0] + glass[1][2] / 2) + glass[0][0] + (glass[0][2] / 2))   # Width diffrance between the glass

            if (DX != 0.0) and (DY != 0.0):                                                 # Make sure the the change happens only if there is an angle
                Theta = math.degrees(math.atan(round(float(DY) / float(DX), 2)))            # Find the Angle
                print "Theta  " + str(Theta)

                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Theta, 1)                 # Find the Rotation Matrix
                Image = cv2.warpAffine(Image, M, (cols, rows))
                # cv2.imshow('ROTATED', Image)                                              # UNCOMMENT IF YOU WANT TO SEE THE

                Face2 = face.detectMultiScale(Image, 1.3, 5)                                # This detects a face in the image
                for (FaceX, FaceY, FaceWidth, FaceHeight) in Face2:
                    CroppedFace = Image[FaceY: FaceY + FaceHeight, FaceX: FaceX + FaceWidth]
                    return CroppedFace


def captureFacePhoto(capture_device, position_ID):
    face_cascade = cv2.CascadeClassifier(configData.HAAR_FACE_CASCADE)
    eye_cascade = cv2.CascadeClassifier(configData.HAAR_EYE_CASCADE)
    time.sleep(1)
    while True:
        # ret,image = capture_device.read()
        image = capture_device.get()
        # image = capture_device.frame.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if np.average(gray)> 100:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)         # Detect the faces and store the positions
            for (x, y, w, h) in faces:                                  # Frames  LOCATION X, Y  WIDTH, HEIGHT
                FaceImage = gray[y - int(h / 2): y + int(h * 1.5),
                            x - int(x / 2): x + int(w * 1.5)]           # The Face is isolated and cropped

                ResultImage = DetectEyes(FaceImage)
                print 'Face detected'
                if ResultImage is not None:
                    frame = ResultImage  # Show the detected faces
                else:
                    frame = gray[y: y + h, x: x + w]

                cv2.imwrite(position_ID+'.jpg',frame)
                img_path = position_ID+'.jpg'
                return (img_path,configData.SUCCESS)
            return configData.NO_FACE_FOUND
        else:
            return configData.BRIGHTNESS_LOW

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

def init_threads(captureDevice):
    videoThread = threading.Thread(target=daemonVideoCapture,args = (captureDevice,buffer_queue,))
    videoThread.setDaemon(True)
    videoThread.start()

    liveStreamThread = threading.Thread(target=showLiveFeed, args=(buffer_queue,))
    liveStreamThread.start()


def Main():
    # from camera import VideoCamera
    captureDevice = createCaptureDevice(0)

    videoThread = threading.Thread(target=daemonVideoCapture,args = (captureDevice,buffer_queue,))
    videoThread.setDaemon(True)
    videoThread.start()

    try:
        ret = create_new_Record('UID_001')
        print ret
    except OSError as e:
        print e
    except Exception as e:
        print e

    t1 = threading.Thread(target=showLiveFeed, args=(buffer_queue,))
    t1.start()
    count = 0
    print 'Ready for capture faces \nPress "c" for Capture'
    while True:

        ch = str(raw_input('Press "c" for capture'))
        if ch == 'c' and count<=10:
            print 'Wait for 1 second for capture of face features'
            path = configData.PARENT_PATH+'UID_001/'+str(count)
            # captureFacePhoto(captureDevice,path)
            ret = captureFacePhoto(buffer_queue,path)
            print ret
            if ret.__contains__('Success'):
                count+=1
        elif count >10 :
            print 'DataSet complete, need only 10 photos'
            return



if __name__=="__main__":
    Main()
    