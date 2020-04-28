import cv2 
import os
  
# Function to extract frames 
def save_frames(frame_path,video_file): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(video_file) 
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count
        #os.mkdir(frame_path+directory) 
        cv2.imwrite(frame_path+"/frame%d.jpg" % count, image) 
  
        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    #save_frames("Path to HMDB action videos ") 
    path="human_action_dataset/"
    directories=os.listdir(path)
    print(directories)
    for directory in directories:
        files=os.listdir(path + directory)
        for file in files:
            frame_path=directory
            try:  
                os.mkdir(frame_path)  
            except OSError as error:  
                print(error)
            print(path+directory+"/"+file)
            save_frames(frame_path,path+directory+"/"+file)
        #    save_frames(path,directory,file)