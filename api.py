import os
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from typing import List, cast
import numpy as np
from pydantic import BaseModel, model_validator
import json
from PIL import Image
import matplotlib.pyplot as plt
import csv
from io import StringIO
import cv2
import random

app = FastAPI()


class BoundingBox(BaseModel):
    left: int
    top: int
    width: int
    height: int

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/get_class_encodings")
async def recognize(
    known_encodings: List[str],
    bounding_boxes_raw: str = Form(),
    class_photo_image: UploadFile = File(...),
):
    print("In")
    bounding_boxes = cast(list[BoundingBox], [BoundingBox.model_validate(bb) for bb in json.loads(bounding_boxes_raw)])
    contents = await class_photo_image.read()  #
    temp_file_name = "class_photos/" + str(random.randint(0, 1000000)) + ".jpg"
    with open(temp_file_name, "wb") as f:
        f.write(contents)
        class_photo = face_recognition.load_image_file(temp_file_name)
    # class_photo = np.frombuffer(contents, np.uint8)
    separate_faces = get_separate_faces(class_photo, bounding_boxes)
    encodings = [encoding_to_string(get_encoding(face)) for face in separate_faces]

    known_encodings_array = [np.array(row.split(","), dtype = float) for row in known_encodings]

    # array of output results of whether student is present is not
    results = []
    for enc in encodings:
        if not enc:
            continue
        uknown_encoding_array = np.array(enc.split(","), dtype=float)
        result = face_recognition.compare_faces(known_encodings_array, uknown_encoding_array)
        if any(result):
            results.append(known_encodings[(result.index(True))])

    return results


@app.post("/add_student")
async def add_student(file: UploadFile = File(...)):
    contents = await file.read()
    temp_file_name = "student_photos/" + str(random.randint(0, 1000000)) + ".jpg"
    with open(temp_file_name, "wb") as f:
        f.write(contents)
        student_photo = face_recognition.load_image_file(temp_file_name)

    # student_photo = np.frombuffer(contents, np.uint8)
    encoding = get_encoding(student_photo)
    print(encoding, type(encoding))

    # Create a unique student_id each time a new student is added

    os.remove(temp_file_name)
    if not encoding:
        return {"encoding": ""}
    return {"encoding": encoding_to_string(encoding) if encoding else ""}


# @app.post("/predict")
# async def predict(known_encodings: List[str], uknown_encoding: str, file: UploadFile = File(...)):
#     known_encodings_array = [np.array(row.split(",")) for row in known_encodings]
#     uknown_encoding_array = np.array(uknown_encoding)
#     results = face_recognition.compare_faces(known_encodings_array, uknown_encoding_array)
#     return results


def encoding_to_string(encoding):
    if not encoding:
        return ""
    encoding = [encoding[0]]

    csv_string_buffer = StringIO()
    csv_writer = csv.writer(csv_string_buffer)
    csv_writer.writerows(encoding)
    return csv_string_buffer.getvalue().strip()


# This is done by the ML Kit I believe? I don't how that will happen
def get_separate_faces(class_photo: np.array, bounding_boxes) -> np.array:
    img = Image.fromarray(class_photo)

    separate_faces = []
    for bound in bounding_boxes:
        left = bound.left
        top = bound.top
        right = left + bound.width
        bottom = top + bound.height
        box = (left, top, right, bottom)
        cropped_img = img.crop(box)
        cropped_array = np.asarray(cropped_img)
        separate_faces.append(cropped_array)

    return separate_faces


def get_encoding(student_photo: np.array) -> np.array:
    # code to get encoding of a face
    return face_recognition.face_encodings(student_photo)


# def find_students(class_photo: np.array) -> List[int]:
#     # Use the bounding boxes to get the images of each student in an array
#     # Get the predictions for each student in an array
#     # See the match from known list and get the name
#     # Finally return the list of all the students detected
#     separate_faces = get_separate_faces(class_photo)

#     # Get this through firebase either here or as a global value
#     known_faces = []
#     predictions = []
#     for face in separate_faces:
#         encoding = get_encoding(face)
#         results = face_recognition.compare_faces(known_faces, encoding)
#         true_indices = [i for i in range(len(results)) if results[i] == True]

#         if len(true_indices) == 0:
#             # Unknown
#             predictions.append(-1)
#             continue
#         elif len(true_indices) > 1:
#             # Multiple hits somehow, don't know what to do :)
#             continue
#         else:
#             # Only one true, good
#             predictions.append(true_indices[0])

#     return predictions

# def get_students_from_index(predictions: List[int]) -> List[str]:
#     # access the
#     return [""]
