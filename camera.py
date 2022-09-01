import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import os
import numpy as np
import argparse


class VideoCamera(object):
    """The class connects to the webcam. And applies the neural network frame by frame to the photo.
     Draws a frame around the face and returns it as a picture. Which enters the stream format"""
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.faceNet = None
        self.maskNet = None
        self.args = None

    def get_frame(self):
        sucess, frame = self.vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = self.detect_and_predict_mask(frame)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def detect_and_predict_mask(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    def construct_model(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("runserver", help='run server')
        ap.add_argument("-f", "--face", type=str,
                        default="face_detector",
                        help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
                        default="mask_detector.model",
                        help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float,
                        default=0.5,
                        help="minimum probability to filter weak detections")

        self.args = vars(ap.parse_args())

        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([self.args["face"],
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        print("[INFO] loading face mask detector model...")
        self.maskNet = load_model(self.args["model"])

        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()


def gen(camera: VideoCamera):
    camera.construct_model()
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
