import face_recognition
import os
import numpy as np

def list_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

# 替换 'your_dataset_directory' 为你的数据集目录
dataset_directory = 'small_dataset'
jpg_files = list_jpg_files(dataset_directory)

unknown_image = face_recognition.load_image_file("unknown6.jpg")   #欲識別的jpg
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]    #進行encoding的特徵抓取
#print(unknown_encoding)

container = np.zeros((len(jpg_files),128))
container_count = 0
for jpg_file in jpg_files:
    known_image = face_recognition.load_image_file(jpg_file)
    known_image_encoding = face_recognition.face_encodings(known_image)[0]
    container[container_count] = known_image_encoding
    container_count+=1

results = face_recognition.api.compare_faces(container, unknown_encoding, tolerance=0.4)
print(results)
print(len(results))
