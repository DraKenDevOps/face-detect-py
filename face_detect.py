import cv2

alg = "haarcascade_frontalface_default.xml"


async def faceDetection(imagePath: str) -> bool:
    image = cv2.imread(imagePath)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image.shape

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + alg)

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(face) == 0:
        return False
    else:
        return True
