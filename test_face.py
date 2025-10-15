import face_recognition

# Load the known image
known_image = face_recognition.load_image_file("known_faces/your_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load an unknown image (you can use the same image to test first)
unknown_image = face_recognition.load_image_file("known_faces/your_face.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces([known_encoding], unknown_encoding)

if results[0]:
    print("✅ Faces match!")
else:
    print("❌ Faces do not match.")
