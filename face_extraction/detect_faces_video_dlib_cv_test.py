from imutils.video import VideoStream
import imutils
from imutils import face_utils
import numpy as np
import argparse
import time
import cv2
from math import floor
import dlib
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", help="path to Caffe 'deploy' prototxt file",
    default="deploy.prototxt")
ap.add_argument("-m", "--model", help="path to Caffe pre-trained model",
    default="res10_300x300_ssd_iter_140000.caffemodel")
ap.add_argument("-c", "--confidence", help="minimum probability to filter weak detections",
    type=float, default=0.6)
ap.add_argument("-v", "--video", default="/home/js8365/data/Sandbox/ClassNSeg/predict.mp4",
    help="video path")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
success, frame = vs.read()
count = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def reshape_box(startX, startY, endX, endY, max_y, max_x):
    w = endX - startX
    h = endY - startY
    cx = startX + (w/2)
    cy = startY + (h/2)

    sq_d = max([h,w]) * 1.2

    n_startX = cx - (sq_d/2)
    n_endX = n_startX + sq_d

    n_startY = cy - (sq_d/2)
    n_endY = n_startY + sq_d    

    if n_startX < 0:
        n_startX = 0
        n_endX = n_startX + sq_d
    if n_startY < 0:
        n_startY = 0
        n_endY = n_startY + sq_d
    if n_endX > max_x:
        n_endX = max_x
        n_startX = n_endX - sq_d
    if n_endY > max_y:
        n_endY = max_x
        n_startY = n_endY - sq_d
    
    return floor(n_startX), floor(n_startY), floor(n_endX), floor(n_endY), floor(cx), floor(cy)


    # loop over the frames from the video stream
while success == True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = imutils.resize(frame, width=500)
    frame_p = frame

    # grab the frame dimensions and convert it to a blob
    (f_h, f_w) = frame.shape[:2]    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(frame, 1)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
     
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        (startX, startY, endX, endY, cx, cy) = reshape_box(x, y, x+w, y+h, f_h, f_w)
        cv2.rectangle(frame_p, (startX, startY), (endX, endY), (0, 255, 0), 2)
     
        # show the face number
        cv2.putText(frame_p, "Dlib #{}".format(i + 1), (startX - 10, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame_p, (x, y), 1, (0, 0, 255), -1)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,(300, 300), (103.93, 116.77, 123.68))
    
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    count = 0    
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        #print(confidence * 100)

 
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue
        count += 1 
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([f_w, f_h, f_w, f_h])
        (startX, startY, endX, endY) = box.astype("int")
        
        dlib_rect = dlib.rectangle(startX, startY, endX, endY)
        shape = predictor(frame, dlib_rect)
        shape = face_utils.shape_to_np(shape)

        (startX, startY, endX, endY, cx, cy) = reshape_box(startX, startY, endX, endY, f_h, f_w)

        
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100) + ", CV #" + str(count)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame_p, (startX, startY), (endX, endY),(255, 255, 0), 2)
        cv2.putText(frame_p, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(frame_p, (x, y), 1, (255, 255, 0), -1)

        # show the output frame
    cv2.imshow("Frame", frame_p)
    key = cv2.waitKey(1) & 0xFF
    
    success, frame = vs.read()
    
    # if the `q` key was pressed, break from the loop
    if (success == False) or (key == ord("q")):
        break
    
# do a bit of cleanup
cv2.destroyAllWindows()