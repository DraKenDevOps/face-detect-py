import cv2
import os
import random

import matplotlib.pyplot as plt

alg = "face-models/haarcascade_frontalface_default.xml"

cwd = os.getcwd()
print("Current working directory =>", cwd)
folder_path = os.path.join(cwd, "images")
print("IMAGE DIR =>", folder_path)

images = []
for filename in os.listdir(folder_path):
    # filepath = os.path.join(folder_path, filename)
    # print(filename)
    # if os.path.isfile(filepath):
    #     with open(filepath, "r") as file:
    #         content = file.read()
    #         print(content)
    images.append("images/" + filename)

randomImg = random.randint(0, len(images))
print("Amount items =>", len(images))
print("Images items =>", images)
print("Image detected =>", randomImg)

# for img in images:
#     print(img)
#     image = cv2.imread(img)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_image.shape
#     face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + alg)
#     face = face_classifier.detectMultiScale(
#         gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
#     )
#     for x, y, w, h in face:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
#     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(img_rgb)
#     plt.show()
#     plt.axis("off")

imagePath = images[randomImg]
image = cv2.imread(imagePath)
# cv2.imshow("Original", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(image.shape)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image.shape

print("cv2.data.haarcascades => ", cv2.data.haarcascades)

# deprecated
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + alg)
face_classifier = cv2.CascadeClassifier(alg)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

if len(face) == 0:
    print({"valid": False})
else:
    print({"valid": True})

for x, y, w, h in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (117, 186, 27), 1)

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.show()
plt.axis("off")
