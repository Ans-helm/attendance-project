import cv2
import face_recognition as fc
import os
import numpy
from datetime import datetime

#Create Database
path = 'student-images'
p_images = []
p_names = []
p_list = os.listdir(path)

for name in p_list:
    this_image = cv2.imread(f'{path}\\{name}')
    p_images.append(this_image)
    p_names.append(os.path.splitext(name)[0])


print(p_names)

#encode images
def encode(images):

    #Encoded list
    encoded_list = []

    #convert all images to rgb
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #ENCODE
        encoded = fc.face_encodings(image)[0]

        #add to the list
        encoded_list.append(encoded)

    #return encoded list
    return encoded_list
def attendance_log(person):
    f = open('register.csv', 'r+')
    data_list = f.readline()
    register_names = []

    for line in data_list:
        new = line.split(',')
        register_names.append(new[0])
    if person not in register_names:
        now = datetime.now()
        string_now = now.strftime('%H:%M:%S')
        f.writelines(f'\n {person}, {string_now}')


encoded_p_list = encode(p_images)

#use webcam
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#read the captured image
success, image = capture.read()

if not success:
    print("Capture could not be taken")
else:
    #read face in camera
    captured_face = fc.face_locations(image)
    #encode capture
    encoded_capture = fc.face_encodings(image, captured_face)
    #search for match
    for face, location_face in zip(encoded_capture, captured_face):
        matches = fc.compare_faces(encoded_p_list, face)
        distances = fc.face_distance(encoded_p_list, face)
        print(distances)
        match_index = numpy.argmin(distances)
        if distances[match_index] > 0.6:
            print("404")
        else:
            #search name
            p_name = p_names[match_index]

            y1, x2, y2, x1 = location_face
            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)
            cv2.rectangle(image,
                          (x1, y2-35),
                          (x2, y2),
                          (0, 255, 0),
                          cv2.FILLED)
            cv2.putText(image,
                        p_name,
                        (x1+6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2)
            attendance_log(p_name)

            #show images
            cv2.imshow('Web Image', image)

            #wait key
            cv2.waitKey(0)






