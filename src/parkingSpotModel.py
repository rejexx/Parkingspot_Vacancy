import os
import numpy as np 
import pandas as pd
import cv2 #open cvs, image processing
from pathlib import Path
from matplotlib import pyplot as plt #image plots
import sys
import streamlit as st
from datetime import datetime, timedelta, timezone
import urllib
import m3u8
import streamlink
import time
import random


######################################
# MaskRCNN config and setup paths
######################################
# Root directory of the project
PROJECT_ROOT = Path("..\\")

# Directory to save logs and trained model (if doing your own training)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on/videos to process
VIDEO_DIR = os.path.join(PROJECT_ROOT, ".\\data\\raw")
# Could use it to train when 'all spots are full' then look for movement

# Video file or camera to process
VIDEO_SOURCE = os.path.join(VIDEO_DIR, "AllSpotsFull_299Frames.ts")

# Local path to output processed videos
VIDEO_SAVE_DIR = os.path.join(PROJECT_ROOT, ".\\data\\processed")
VIDEO_SAVE_FILE = os.path.join(VIDEO_SAVE_DIR, "FinalFile.avi")


######################################
# Functions
######################################

def display_video(image_placeholder, video_file):
    """Shows a video in given streamlit placeholder image
    image_placeholder - an st.empty streamlit object
    video_file - string path to video, entire video will be shown"""
    
    # Load the video file we want to display
    video_capture = cv2.VideoCapture(video_file)
    
    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        image_placeholder.image(frame, channels="BGR")
        time.sleep(0.01)

    # Clean up everything when finished
    video_capture.release()


def display_single_frame(video_file, image_placeholder, frame_index=0):
        """Displays a single frame in streamlit
            Inputs:
            video_file - path to file name or other openCV video
            image_placeholder - streamlit st.empty() or image object
            frame_index - frame number to show, 0 indexed. Do not exceed max frames
            """
        video_capture = cv2.VideoCapture(video_file)

        video_capture.set(1, frame_index)
        success, frame = video_capture.read()
        
        image_placeholder.image(frame, channels="BGR")


        video_capture.release()


def main():    
    ######################################
    # Streamlit
    ######################################
    st.title("Parking Spot Finder")
    
    tempVideo = ""
    if st.button('Get live clip'):
        #Temp file to store latest clip in, should delete these later.
        tempVideo = os.path.join(VIDEO_DIR,f"{datetime.now().strftime('%m_%d_%Y %H-%M')}.ts")  #files are format ts, open cv can view them
    
        st.write(f"Live video at {datetime.now()}....")
    
        #Get a video clip
        videoURL = "https://youtu.be/DoUOrTJbIu4" #Jackson hole town square, live stream
        
    
        #Get the video
        st.write(f"Getting the video from youtube: {videoURL}:")
        dl_stream(videoURL, tempVideo, 1)
    
        # Load the video file we want to display
        video_capture = cv2.VideoCapture(tempVideo)
    
        #streamlit placeholder for image/video
        image_placeholder= st.empty()
    
        # Loop over each frame of video
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
    
            image_placeholder.image(frame, channels="BGR")
            time.sleep(0.01)
    
        # Clean up everything when finished
        video_capture.release()
    
    if st.button('Show saved clip'):
    #Temp file to store latest clip in, should delete these later.
        st.write(f"Video: {(VIDEO_SOURCE)}")
        
        totalFrames = frame_count(VIDEO_SOURCE, manual=True) - 1
        
        frame_index = st.slider(label="Show frame:", min_value=0, max_value=totalFrames, value=0,
                  step=1, key="savedClipFrame", help="Choose frame to view")
                  
        #streamlit placeholder for image/video
        image_placeholder= st.empty()

        # Load the video file we want to display
        display_single_frame(VIDEO_SOURCE, image_placeholder, frame_index)
    
    
    #Check for spots on temp file
    if st.button("Process video clip"):
        
        # Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
        ROOT_DIR = 'aktwelve_Mask_RCNN'
        assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'
        #sys.path.append("aktwelve_Mask_RCNN")
        sys.path.append(ROOT_DIR)
        # Import mrcnn libraries
        import mrcnn.config
        import mrcnn.utils
        from mrcnn.model import MaskRCNN
    
            
        dl_weights_warning = st.warning("Getting COCO trained weights file")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            dl_weights_warning.warning("Downloading COCO weights. This may take a while")
            mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)
        dl_weights_warning.empty()
    
        #Give message while loading weights
        weight_warning = st.warning("Loading model weights, hold on...")
    
        
        #@st.cache()  #This is mutating, and causes issues.  Would be nice to fix
        def maskRCNN_model(model_dir, trained_weights_file):     
            # Configuration that will be used by the Mask-RCNN library
            class MaskRCNNConfig(mrcnn.config.Config):
                NAME = "coco_pretrained_model_config"
                IMAGES_PER_GPU = 1
                GPU_COUNT = 1
                NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
                DETECTION_MIN_CONFIDENCE = 0.6
            
            # Create a Mask-RCNN model in inference mode
            model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
        
            
            # Load pre-trained model
            model.load_weights(trained_weights_file, by_name=True)
            model.keras_model._make_predict_function()
            
            return model
        
        #Create model with saved weights
        model = maskRCNN_model(model_dir=MODEL_DIR, trained_weights_file=COCO_MODEL_PATH)
        
        weight_warning.empty()  #Make the warning go away, done loading
        
        #load or create bounding boxes
        boundingBoxFile = os.path.join(PROJECT_ROOT,'data\processed\spot_boxes_6.csv')
    
        #Load boxes from file if they exist
        #Else process the saved file and make boxes from cars that don't move.
        if os.path.exists(boundingBoxFile):
        #if False:  #Force re-run
            parked_car_boxes = np.loadtxt(boundingBoxFile, dtype='int', delimiter=',')
            st.write(f"Loaded {len(parked_car_boxes)} existing bounding boxes from file")
        else:
            #Learn where boxes are from movie, and save video with annotations
            #Sources is either VDIEO_SOURCE or try with tempFile
            computeBoxes_warning= st.warning("No saved bounding boxes - will process to make new ones")
            # detectSpots(video_file, video_save_file='findParkingSpaces.avi', model, utils, initial_check_frame_cutoff=10):
            parked_car_boxes = detectSpots(VIDEO_SOURCE, video_save_file=os.path.join(VIDEO_SAVE_DIR, 
                "findSpots3.avi"), model=model, utils=mrcnn.utils, initial_check_frame_cutoff=299)
            
            #One of those 'spots' is actually a car on the road, I'm going to remove it manually
            badSpot = np.where(parked_car_boxes == [303,   0, 355,  37])
            parked_car_boxes = np.delete(parked_car_boxes, badSpot[0][0], axis=0)
            
            #Save edited boxes to file for future use
            np.savetxt(boundingBoxFile, parked_car_boxes, delimiter=',')
            computeBoxes_warning.empty()
            st.write(f"Saved new bounding boxes to: {boundingBoxFile}")
            
        #if we didn't get a new image, load one from file
        if (not os.path.exists(tempVideo) or tempVideo == ""):
            tempVideo= os.path.join(VIDEO_DIR,"05_19_2021 14-20.ts")
    
        # Process video, outputs a image to streamlit
        # This is where the real work actually happens
        st.write(f"Looking at file: {tempVideo}")
        countSpots_warning = st.warning("Counting spots in video")
        # countSpots(video_source, parked_car_boxes, model, utils, video_save_file="annotatedVideo.avi",
        #       framesToProcess=True, freeSpaceFrameCutOff=5, showVideo = True, skipNFrames = 1)
        vacancyPerFrame = countSpots(video_source = tempVideo, 
                            parked_car_boxes=parked_car_boxes,
                            model=model, 
                            utils = mrcnn.utils,
                            video_save_file=VIDEO_SAVE_FILE, 
                            freeSpaceFrameCutOff=0, 
                            skipNFrames=10)
        
        countSpots_warning.empty()  #Clear the warning/loading message
    
        vacancyPerFrame_df = pd.DataFrame(vacancyPerFrame, index=["Available spots"]).T
        vacancyPerFrame_df.index.name = "Frame number"
        st.bar_chart(data=pd.DataFrame(vacancyPerFrame_df))
        
        st.write(f"Spaces available by frame: {vacancyPerFrame}")
        

# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    '''Filter a list of Mask R-CNN detection results 
        to get only the detected cars / trucks
        @boxes - bounding boxs
        @class_ids - the matterport class ID
        Return: bounding boxes that are cars'''
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


def writeFramesToFile(frame_array, fileName="video.avi", nthFrames=1, fps=15):
    '''writeFramesToFile(frame_array, fileName="video.avi", nthFrames=1, fps=15)
    Writes array of images to a video file of type .avi
    parameters:
      frame_array - python list of frames as pixel values from openCV
      fileName - path to save file
      nthFrames - how many frames to keep, 1 will keep all frames, 2 will remove every other, etc...
      fps - frames per second'''
    assert (len(frame_array) > 0), "No images to save, frame_array is empty"

    #check first frame and find shape
    width = len(frame_array[0][0])
    height = len(frame_array[0])
    size = (width,height)

    # make video writer
    out = cv2.VideoWriter(fileName,cv2.VideoWriter_fourcc(*'DIVX'), fps, size) #name, writer, fps, size

    for i in range(0, len(frame_array), nthFrames):
        out.write(frame_array[i])
    out.release()
    return None


# Ref: https://stackoverflow.com/questions/55631634/recording-youtube-live-stream-to-file-in-python
def get_stream(url):
    """
    Get upload chunk url
    input: youtube URL
    output: m3u8 object segment
    """
    
    #Try this line tries number of times, if it doesn't work, 
    # then show the exception on the last attempt
    # Credit, theherk, https://stackoverflow.com/questions/2083987/how-to-retry-after-exception
    tries = 10
    for i in range(tries):
        try:
            streams = streamlink.streams(url)
        except:
            if i < tries - 1: # i is zero indexed
                print(f"Attempt {i+1} of {tries}")
                time.sleep(0.01) #Wait half a second
                continue
            else:
                raise
        break
        
    #print(f"Stream choices: {streams.keys()})
    stream_url = streams["360p"] #Alternate, use "best"

    m3u8_obj = m3u8.load(stream_url.args['url'])
    return m3u8_obj.segments[0] #Parsed stream


def dl_stream(url, filename, chunks):
    """
    Download each chunk to file
    input: url, filename, and number of chunks (int)
    output: saves file at filename location
    returns none.
    """
    pre_time_stamp = datetime(1, 1, 1, 0, 0, tzinfo=timezone.utc)
    #Repeat for each chunk
    #Needs to be in chunks beceause 
    #  1) it's live and 
    #  2) it won't let you leave the stream open forever
    i=1
    while i <= chunks:
       
        #Open stream
        stream_segment = get_stream(url)
        
        #Get current time on video
        cur_time_stamp = stream_segment.program_date_time
        #Only get next time step, wait if it's not new yet
        if cur_time_stamp <= pre_time_stamp:
            #Don't increment counter until we have a new chunk
            print("NO   pre: ",pre_time_stamp, "curr:",cur_time_stamp)
            time.sleep(0.5) #Wait half a sec
            pass
        else:
            print("YES: pre: ",pre_time_stamp, "curr:",cur_time_stamp)
            print(f'#{i} at time {cur_time_stamp}')
            #Open file for writing stream
            file = open(filename, 'ab+') #ab+ means keep adding to file
            #Write stream to file
            with urllib.request.urlopen(stream_segment.uri) as response:
                html = response.read()
                file.write(html)
                
            #Update time stamp
            pre_time_stamp = cur_time_stamp
            time.sleep(stream_segment.duration-1) #Wait duration time - 1

            i += 1 #only increment if we got a new chunk
        file.close()
    return None


def frame_count(video_path, manual=False):
    """frame_count - get how many frames are in a video
    video_path - path to video
    manual - True or False, manual method is much more accurate but slow"""
    #Credit: https://stackoverflow.com/questions/25359288/
    #        how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    # answer from: nanthancy 
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method 
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    return frames


def countSpots(video_source, parked_car_boxes, model, utils, video_save_file="annotatedVideo.avi",
               framesToProcess=True, freeSpaceFrameCutOff=5, showVideo = True, skipNFrames = 1):
    '''Counts how many spots are vacant at the end of the video 
    saves a video showing spots being vacant
    returns: count of spots in final frame that are vacant
    inputs: video_source: file path
            parked_car_boxes: bounding boxes of parking spaces
            model: Mask R-CNN inference model object
            video_save_file: default "annotatedVideo.avi", video with annotations
            framesToProcess: default to all frames.  Enter an int to process only part of the file. Good for saving time
            freeSpaceFrameCutOff: default 2, number of frames a spot must be empty before appearing as such, helps with jitter'''
    
    assert (skipNFrames > 0), "skipNFrames must be greater than 0. Default is 1 (no skipping)"
    assert (freeSpaceFrameCutOff >= 0), "freeSpaceFrameCutOff can't be negative. Default is 5"
    
    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(video_source)

    #Store the annotated frames for output to video
    frame_array=[]
    
    #Speed processing by skipping n frames, so we need to keep track
    frameNum = 0
    
    #How many free spots per frame, #frame_Num:vacant spots
    vacancyPerFrame = {} 
    
    #Dictionary of parking space index and how many frames it's been 'free'
    carBoxes_OpenFrames = {i: 0 for i in range(len(parked_car_boxes))}
    image_placeholder_processing= st.empty()

    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        
        #Skip every nth frame to speed processing up
        if (frameNum % skipNFrames != 0):
            frameNum += 1
            pass
        else:
            frameNum += 1

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]

            print(f"Processing frame: #{frameNum}")
            # Run the image through the Mask R-CNN model to get results.
            results = model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # The r variable will now have the results of detection:
            # - r['rois'] are the bounding box of each detected object
            # - r['class_ids'] are the class id (type) of each detected object
            # - r['scores'] are the confidence scores for each detection
            # - r['masks'] are the object masks for each detected object (which gives you the object outline)

            # We already know where the parking spaces are. Check if any are currently unoccupied.

            # Get where cars are currently located in the frame
            car_boxes = get_car_boxes(r['rois'], r['class_ids'])

            # See how much those cars overlap with the known parking spaces
            overlaps = utils.compute_overlaps(parked_car_boxes, car_boxes)

            # Assume no spaces are free until we find one that is free
            free_spaces = 0

            # Loop through each known parking space box
            for row, areas in enumerate(zip(parked_car_boxes, overlaps)):
                parking_area, overlap_areas = areas
                # For this parking space, find the max amount it was covered by any
                # car that was detected in our image (doesn't really matter which car)
                max_IoU_overlap = np.max(overlap_areas)

                # Get the top-left and bottom-right coordinates of the parking area
                y1, x1, y2, x2 = parking_area

                # Check if the parking space is empty by seeing if any car overlaps
                # it by more than x amount using IoU
                if max_IoU_overlap < 0.20:    
                    #If the spot has appeared open long enough, count it as free!
                    # This is so we don't alert based on one frame of a spot being open/closed.
                    # This helps prevent the script triggered on one bad detection.
                    if carBoxes_OpenFrames[row]+1 >= freeSpaceFrameCutOff:
                        # Parking space not occupied! Draw a green box around it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                        #Count this as occupied, and don't let it be immediately changed
                        carBoxes_OpenFrames[row] = max(freeSpaceFrameCutOff,1)
                    else:
                        # Parking space hasn't been vacant long enough - draw a red box around it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                        # Tag this spot as being open for +1 frame
                        carBoxes_OpenFrames[row] += 1

                else:
                    #else, spot appears occupied this frame 
                    #If it's been occupied for more than the frame cutoff:
                    if carBoxes_OpenFrames[row] <= freeSpaceFrameCutOff:
                        # Parking space is occupied - draw a red box around it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        #Set the frame counter to 0, it's full, don't let it change immediately
                        carBoxes_OpenFrames[row] = 0 
                    else:
                         #Start counting frames this spot is full,
                        # So script isn't triggered from someone driving by
                        carBoxes_OpenFrames[row] -= 1

                        # Parking space still 'free'. Draw a green box around it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Write the IoU measurement inside the box
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

            # If a space has been free for several frames, let's count it as free
            #  loop through all 'free' frames and sum the result
            free_spaces = sum([int(i) > freeSpaceFrameCutOff for i in carBoxes_OpenFrames.values()])
            
            #Save num free spaces in frame to dict for final output
            vacancyPerFrame[frameNum] = free_spaces


            # Write how many free spots there are at the top of the screen
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"Free Spaces: {free_spaces}", (30, 30), font, 1.0, (0, 255, 0), 2, cv2.FILLED)

            print(f'Free Spaces: {free_spaces}')
            #print number of frames
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, f"Frame: {frameNum}", (10, 340), font, 0.5, (255, 255, 255))

            # Show the video in a new window
            if showVideo:
                image_placeholder_processing.image(frame, channels="BGR")
                time.sleep(0.01)

            #Append frame to outputvideo
            frame_array.append(frame)
            
            if (framesToProcess != True and frameNum > framesToProcess):
                print(f"Stopped processing at frame {frameNum} as requested by framesToProcess parameter")
                break

    # Clean up everything when finished
    video_capture.release()  #free the video
    if showVideo: 
        cv2.destroyAllWindows()  #Close the video player
    writeFramesToFile(frame_array=frame_array, fileName=video_save_file) #save the file
       
    st.write("done!")
    return vacancyPerFrame


def detectSpots(video_file, model, utils, video_save_file='findParkingSpaces.avi', showVideo=True, initial_check_frame_cutoff=10):
    '''detectSpots(video_file, initial_check_frame_cutoff=10)
    Returns: np 2D array of bounding boxes of all bounding boxes that are still occupied
    after initial_check_frame_cutoff frames.  These can be considered "parking spaces".
    
    An update might identify any spaces that get occupied at some point and stay occupied 
    for a set length of time, in case some areas start off vacant.'''
    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(video_file)

    #Store the annotated frames for output to video/counting how many frames we've seen
    frame_array=[]

    #Will contain bounding boxes of parked cars to identify 'parkable spots'
    parked_car_boxes = []
    parked_car_boxes_updated = []
    
    #Make image appear in streamlit
    image_placeholder_processing= st.empty()
    
    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            st.write("Processed {len(frame_array)} frames of video, exiting.")
            return parked_car_boxes

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = frame[:, :, ::-1]

        #ignore the inbetween frames 0 to x, don't run the model on them and save processing time
        if 0 < len(frame_array) < initial_check_frame_cutoff: 
            print(f"ignore this frame for processing, #{len(frame_array)}")
        else:
            print(f"Processing frame: #{len(frame_array)}")
            # Run the image through the Mask R-CNN model to get results.
            #model.keras_model._make_predict_function() #Will this solve my bug?
            #Or try this next https://stackoverflow.com/questions/54652536/keras-tensorflow-backend-error-tensor-input-10-specified-in-either-feed-de

            results = model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # The r variable will now have the results of detection:
            # - r['rois'] are the bounding box of each detected object
            # - r['class_ids'] are the class id (type) of each detected object
            # - r['scores'] are the confidence scores for each detection
            # - r['masks'] are the object masks for each detected object (which gives you the object outline)

            if len(frame_array) == 0:
                # This is the first frame of video,
                # Save the location of each car as a parking space box and go to the next frame of video.
                # We check if any of those cars moved in the next 5 frames and assume those that don't are parked
                parked_car_boxes =  get_car_boxes(r['rois'], r['class_ids'])
                parked_car_boxes_init = parked_car_boxes
                print('Parking spots 1st frame:', len(parked_car_boxes))

            #If we are past the xth initial frame, already know where parked cars are, then check if any cars moved:                                        
            else:
                # We already know where the parking spaces are. Check if any are currently unoccupied.

                # Get where cars are currently located in the frame
                car_boxes = get_car_boxes(r['rois'], r['class_ids'])

                # See how much those cars overlap with the known parking spaces
                overlaps = utils.compute_overlaps(parked_car_boxes, car_boxes)

                # Loop through each known parking space box
                for row, areas in enumerate(zip(parked_car_boxes, overlaps)):
                    parking_area, overlap_areas = areas
                    # For this parking space, find the max amount it was covered by any
                    # car that was detected in our image (doesn't really matter which car)
                    max_IoU_overlap = np.max(overlap_areas)

                    # Get the top-left and bottom-right coordinates of the parking area
                    y1, x1, y2, x2 = parking_area

                    # Check if the parking space is occupied by seeing if any car overlaps
                    # it by more than x amount using IoU
                    if max_IoU_overlap < 0.20:
                        #In the first few frames, remove this 'spot' and consider it as a moving car instead
                        # Transient event, draw green box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    else:
                        #if len(frame_array) == initial_check_frame_cutoff: 
                        #Consider this a parking spot, car is still in it!
                        #Dangerous to mutate array while using it! So make a new one
                        parked_car_boxes_updated.append(list(parking_area))

                        # Parking space is still occupied - draw a red box around it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                    # Write the top and bottom corner locations in the box for ref
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, str(parking_area), (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

                parked_car_boxes = np.array(parked_car_boxes_updated)  #only happens once

        #print number of frames
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, f"Frame: {len(frame_array)}", (10, 340), font, 0.5, (0, 255, 0), 2, cv2.FILLED)

        # Show the frame of video on the screen
        if showVideo:
            image_placeholder_processing.image(frame, channels="BGR")
            time.sleep(0.01)

        #Append frame to outputvideo
        frame_array.append(frame)

        #stop when cutoff reached
        if len(frame_array) > initial_check_frame_cutoff:
            print(f"Finished, processed frames: 0 - {len(frame_array)}")
            break;
        

    # Clean up everything when finished
    video_capture.release()
    writeFramesToFile(frame_array=frame_array, fileName=video_save_file)

    #Show final image in matplotlib for ref
    return parked_car_boxes


if __name__ == "__main__":
    main()

