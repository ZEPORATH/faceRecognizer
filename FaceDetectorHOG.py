import datetime
import argparse
import numpy as np
import dlib
import urllib2
import cv2
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
smooth_fps = 0
# cap = cv2.VideoCapture(0)
# Replace the URL with your own IPwebcam shot.jpg IP:port
# host = "192.168.0.100:8080"
# if len(sys.argv)>1:
#     host = sys.argv[1]
#
# hoststr = 'http://' + host + '/video'
# print 'Streaming ' + hoststr
#
# stream=urllib2.urlopen(hoststr)


def detect_face_from_IPCamera(host_address):
    host_string = 'http://'+host_address+'/video'
    print 'Streaming '+ host_string
    stream = urllib2.urlopen(host_string)
    bytes = ''
    smooth_fps = 0
    while True:
        bytes += stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)

            result_frame,smooth_fps = detect_faces(image,smooth_fps)
            cv2.namedWindow("Face", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("Face", result_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def sharpen_image(frame,typeof):
    # generating the kernels
    # kernel_sharpen_1 = simple_sharpening
    # kernel_sharpen_2 = excessive_sharpening
    # kernel_sharpen_3 = Edge_sharpening
    # case 4 = Gaussian Unsgarp Masking, param, plays a role here
    #        use of case 4 , should be avoided, it's complete implementation is pending
    kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    img = frame
    if typeof == 1:
        output1 = cv2.filter2D(img, -1, kernel_sharpen_1)
        return output1
    elif typeof == 2:
        output2 = cv2.filter2D(img, -1, kernel_sharpen_2)
        return output2
    elif typeof == 3:
        output3 = cv2.filter2D(img, -1, kernel_sharpen_3)
        return output3
    elif typeof == 4:
        # Unsharp Mask technique
        gaussian3 = cv2.GaussianBlur(img, (9, 9), 10.0)
        unsharp_img = cv2.addWeighted(img, 1.6, gaussian3, -0.5, 0, img)
        return unsharp_img

def detectFacesFromCamera(source = 0):
    cap = cv2.VideoCapture(source)
    smooth_fps = 0

    while True:
        start_time = cv2.getTickCount()

        ret,image = cap.read()
        # image = cv2.resize(image, (640,480))
        # image = sharpen_image(image,4)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = sharpen_image(gray,4)
        rects = detector(gray,0)
        cv2.namedWindow("Face", cv2.WND_PROP_FULLSCREEN)
        for rect in rects:
            shape = predictor(gray, rect)
            # print type(shape),shape
            shape = shape_to_np(shape)


            x1 = 0;
            y1 = 0;
            count = 0
            for (x, y) in shape:
                if count == 0:
                    x1 = x;
                    y1 = y
                    count += 1
                else:
                    cv2.line(image, (x1, y1), (x, y), (0, 0, 255), 1)
                    x1 = x;
                    y1 = y
                    # count+=1
        end_time = cv2.getTickCount()

        fps = cv2.getTickFrequency() / (end_time - start_time)
        smooth_fps = 0.9 * smooth_fps + 0.1 * fps

        print("FPS:", smooth_fps)

        cv2.imshow("Face", image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()

def detect_faces(frame,smooth_fps):

    start_time = cv2.getTickCount()
    image = frame
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # cv2.namedWindow("Face", cv2.WND_PROP_FULLSCREEN)
    for rect in rects:
        shape = predictor(gray, rect)
        # print type(shape),shape
        shape = shape_to_np(shape)

        # print type(shape), len(shape)
        # for (x, y) in shape:
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        x1 = 0;
        y1 = 0;
        count = 0
        for (x, y) in shape:
            if count == 0:
                x1 = x;
                y1 = y
                count += 1
            else:
                cv2.line(image, (x1, y1), (x, y), (0, 0, 255), 1)
                x1 = x;
                y1 = y
                # count+=1
    end_time = cv2.getTickCount()

    fps = cv2.getTickFrequency() / (end_time - start_time)
    smooth_fps = 0.9 * smooth_fps + 0.1 * fps

    print("FPS:", smooth_fps)

    # cv2.imshow("Face", image)
    return image,smooth_fps
if __name__=="__main__":
    detectFacesFromCamera(source=0)
    # detect_face_from_IPCamera('192.168.0.101:8080')

cv2.destroyAllWindows()
