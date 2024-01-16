import numpy as np
import cv2
import random
import os
# import matplotlib.pyplot as plt


protxt_path = "./face-models/deploy.prototxt"
caffemodel_path = "./face-models/res10_300x300_ssd_iter_140000.caffemodel"
cwd = os.getcwd()
folder_path = os.path.join(cwd, "images")
images = []
for filename in os.listdir(folder_path):
    images.append("images/" + filename)

randomImg = random.randint(0, len(images))
print("Amount items =>", len(images))
print("Images items =>", images)
print("Image detected =>", randomImg)


def readFaceNet(prototxt, caffemodel):
    faceNet = cv2.dnn.readNet(prototxt, caffemodel)
    return faceNet


def detectFace(image, FaceNet, confidence_input):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    FaceNet.setInput(blob)
    detections = FaceNet.forward()
    print("Detections => ", detections)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_input:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi = image[startY:endY, startX:endX].copy()
            return roi, confidence
    return None, None


class FaceDetection:
    def __init__(self, protxt_path, caffemodel_path):
        self.protxt = protxt_path
        self.caffemodel = caffemodel_path

    def detect_face(self, image, confidence):
        faceNet = readFaceNet(self.protxt, self.caffemodel)
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            image = image
        detected_face, confidence = detectFace(image, faceNet, confidence)
        return detected_face, confidence


facedetection = FaceDetection(protxt_path, caffemodel_path)
face = facedetection.detect_face(images[1], 0.1)
# image = cv2.imread(images[1])
# for x, y, w, h in face[0]:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (117, 186, 27), 1)
# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20, 10))
# plt.imshow(img_rgb)
# plt.show()
# plt.axis("off")
cv2.imshow("Result " + images[1], face[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
