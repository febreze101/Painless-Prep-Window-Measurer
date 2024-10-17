import os
import cv2

cwd = os.getcwd()

vid_path = os.path.join(cwd, 'vids', 'file_example_MP4_640_3MG.mp4')

video = cv2.VideoCapture(vid_path)

ret = True

while ret:
    ret, frame = video.read()
    
    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(40)
        
video.release()