import Trainer_All
import configData
import os
import cv2
##Takes path to the dataset
## For eg: /dataSets/
path = configData.PARENT_PATH
def generate_trainer(source = path):
    IDs, FaceList = Trainer_All.getImageWithID(source)
    # print IDs, FaceList
    EigenFace,LBPHFace = Trainer_All.train_all_recogniser(IDs,FaceList)
    return EigenFace,LBPHFace


def DetectEyes(Image):
    import math
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


def generate_faces(source):
    import glob
    import numpy as np
    import math
    face_cascade = cv2.CascadeClassifier(configData.HAAR_FACE_CASCADE)

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print imagePaths

    for imagePath in imagePaths:
        for filename in os.listdir(imagePath):
            img_path = imagePath+os.sep+filename
            img_path = os.path.abspath(img_path)
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if np.average(gray) > 100:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect the faces and store the positions
                for (x, y, w, h) in faces:  # Frames  LOCATION X, Y  WIDTH, HEIGHT
                    FaceImage = gray[y - int(h / 2): y + int(h * 1.5),
                                x - int(x / 2): x + int(w * 1.5)]  # The Face is isolated and cropped

                    ResultImage = DetectEyes(FaceImage)
                    print 'Face detected'
                    if ResultImage is not None:
                        frame = ResultImage  # Show the detected faces
                    else:
                        frame = gray[y: y + h, x: x + w]
                    filename.replace('.jpg','c.jpg')
                    cv2.imwrite(img_path, frame)
                    # img_path = position_ID + '.jpg'
                    print (1, configData.SUCCESS)
                print configData.NO_FACE_FOUND
            else:
                print configData.BRIGHTNESS_LOW

def Main():
    # generate_faces(path)
    generate_trainer(path)
if __name__=="__main__":
    Main()