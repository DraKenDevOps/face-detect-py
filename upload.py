import os
from fastapi import FastAPI, UploadFile, File
from secrets import token_hex
import face_detect
from fastapi.staticfiles import StaticFiles

cwd = os.getcwd()
# upload_path = os.path.join(cwd, "uploads")

app = FastAPI()

app.mount("/static", StaticFiles(directory="uploads"), name="static")


@app.post("/upload")
async def save_upload_file(image_file: UploadFile = File(...)) -> dict:
    file_ext = image_file.filename.split(".").pop()  # .png ...
    file_name = token_hex(10)
    file = f"{file_name}.{file_ext}"

    file_path = os.path.join(cwd, "uploads", file)

    print("Upload to =>", file_path)

    with open(file_path, "wb") as f:
        content = await image_file.read()
        f.write(content)

    try:
        check = await face_detect.faceDetection(file_path)
        return {"valid": check, "image": f"{file}"}
    except Exception as inst:
        print(inst)
