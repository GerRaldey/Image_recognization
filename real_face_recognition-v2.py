# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:57:15 2020

@author: Gerald
"""

import cv2
import face_recognition
import os
from datetime import date 
import time


start_time = time.time()

webcam_video_stream = cv2.VideoCapture(0)

#load the sample images and get the 128 face embeddings from them
#Sample image for Modi

directory = "F:\\Programming Lessons\\Python Facial Recognition\\images"

known_face_encodings = []

for filename in os.listdir(directory):
    face_images = face_recognition.load_image_file("F:\\Programming Lessons\\Python Facial Recognition\\images\\" + filename)
    encoded_images = face_recognition.face_encodings(face_images)[0]
    known_face_encodings.append(encoded_images)
        
        
known_face_names = ["Alfred Pepito", "Arielle Famador", "Chrsitopher Melbacuno", "Darel Diano", "Dindo Labastida", "Donald Trump", "Dwyane Wade", "Evan Compuesto", "Gabriel Arrabis", "George Ligan", "Gerald Minoza", "Herald Pepito", "Jason Statham", "Jayden Bargayo", "Jade Minoza", "John Christian Velasquez", "Josua Oswa", "Kamala Harris", "Kevin Ace", "Khenyl Lopez", "Kurt lao", "Lebron James", "Marvin Nunez", "Narendra Modi", "Norchri Conjie", "Oliver Cuyos", "Renero Codilla", "Ritzniel Aragon", "Rowan Atkiinson", "Sergio Taghoy"]

all_face_locations = []    
all_face_encodings = [] 
all_face_names = []    



#loop through every frame in the video
matching = True
while matching:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
    
    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)


    #looping through the face locations and the face embeddings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        status = matching
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        #change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
       
        #string to hold the label
        name_of_person = 'Unknown face'
        
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            print("Face recognized successfully! Name: ")
            print("Date: ", date.today().strftime("%b. %d, %Y"))
            break
        else:
            print("Your face didn't match your account ID. Please try again...")
            print("Date: ", date.today().strftime("%b. %d, %Y"))
            matching = False
            
            
            
        #draw rectangle around the face    
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
    #display the video
    cv2.imshow("Webcam Video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 
end_time = time.time()
print("Time consumed: ", (end_time - start_time), " seconds.")

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()      


