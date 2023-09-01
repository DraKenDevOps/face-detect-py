import os
import cv2
import uuid

alg = "haarcascade_frontalface_default.xml"


async def faceDetection(imagePath: str) -> bool:
    cwd = os.getcwd()
    image = cv2.imread(imagePath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image.shape

    face_model = os.path.join(cv2.data.haarcascades, alg)

    face_classifier = cv2.CascadeClassifier(face_model)

    print(face_classifier.empty())

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    file_path = os.path.join(cwd, "detected", f"Detected-{uuid.uuid4()}.jpg")

    print("Detected =>", file_path)

    # cv2.imwrite(file_path, face)

    if len(face) == 0:
        return False
    else:
        return True
