from io import StringIO
import face_recognition
import numpy as np
import csv

# Load the jpg files into numpy arrays
virat_image = face_recognition.load_image_file("training/p1/IMG-20240508-WA0076.jpg")
garg_image = face_recognition.load_image_file("training/p2/IMG-20240508-WA0062.jpg")
pulkit_image = face_recognition.load_image_file("training/p3/IMG-20240508-WA0083.jpg")
gurnoor_image = face_recognition.load_image_file("training/p4/IMG-20240508-WA0070.jpg")
shivansh_image = face_recognition.load_image_file("training/p5/IMG-20240508-WA0081.jpg")
shreyas_image = face_recognition.load_image_file("training/p6/WhatsApp Image 2024-05-08 at 13.19.28_0d9f9292.jpg")

unknown_image = face_recognition.load_image_file("test/werat.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    virat_face_encoding = face_recognition.face_encodings(virat_image)[0]
    garg_face_encoding = face_recognition.face_encodings(garg_image)[0]
    pulkit_face_encoding = face_recognition.face_encodings(garg_image)[0]
    gurnoor_face_encoding = face_recognition.face_encodings(garg_image)[0]
    shivansh_face_encoding = face_recognition.face_encodings(garg_image)[0]
    shreyas_face_encoding = face_recognition.face_encodings(garg_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    virat_face_encoding,
    garg_face_encoding,
    pulkit_face_encoding,
    gurnoor_face_encoding,
    shivansh_face_encoding,
    shreyas_face_encoding,
]

# Create a StringIO StringIOo act as an in-memory file
csv_string_buffer = StringIO()

# Create a csv writer object using the StringIO buffer
csv_writer = csv.writer(csv_string_buffer)

# Write the array data to the CSV writer
csv_writer.writerows([virat_face_encoding.tolist()])

# Get the CSV data as a string
csv_string = csv_string_buffer.getvalue()

# The csv_string now contains your NumPy array data in CSV format
print(csv_string)

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
print(results.index(True))

print("Is the unknown face a picture of Virat? {}".format(results[0]))
print("Is the unknown face a picture of Garg? {}".format(results[1]))
print("Is the unknown face a picture of Pulkit? {}".format(results[2]))
print("Is the unknown face a picture of Gurnoor? {}".format(results[3]))
print("Is the unknown face a picture of Shivansh? {}".format(results[4]))
print("Is the unknown face a picture of Shreyas? {}".format(results[5]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
