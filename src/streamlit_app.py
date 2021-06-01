# IMPORTS
# general
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

# Video getting and saving
import cv2  # open cvs, image processing
import urllib
import m3u8
import time
import pafy  # needs youtube_dl

# File handling
from pathlib import Path
import os
import pickle
from io import BytesIO
import requests
import sys

# Import mrcnn libraries

######################################
# MaskRCNN config and setup paths
######################################
# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
#ROOT_DIR = r'https://raw.githubusercontent.com/akTwelve/Mask_RCNN/master'
#ROOT_DIR = 'aktwelve_Mask_RCNN'
#assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'
#sys.path.append("aktwelve_Mask_RCNN")
#sys.path.append(ROOT_DIR)

import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN


# Root directory of the project
PROJECT_ROOT = Path(r"https://raw.githubusercontent.com/rejexx/Parkingspot_Vacancy/main/")

# Directory to save logs and trained model (if doing your own training)
MODEL_DIR = r"C:\springboard\Parkingspot_Vacancy\models"

# Root directory of the project
PROJECT_ROOT = Path("..\\")

# Directory to save logs and trained model (if doing your own training)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# path to trained weights file
COCO_MODEL_PATH = r"https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5" #"mask_rcnn_coco.h5" # local
#COCO_MODEL_PATH ="mask_rcnn_coco.h5" # local

# Preprocessed demo video
DEMO_VIDEO = r"https://github.com/rejexx/Parkingspot_Vacancy/blob/main/src/streamlit_app/demo.avi?raw=true"

# Local path to output processed videos
VIDEO_SAVE_DIR = os.path.join(PROJECT_ROOT, ".\\data\\processed")
VIDEO_SAVE_FILE = os.path.join(VIDEO_SAVE_DIR, "FinalFile.avi")


######################################
# Functions
######################################


def main():
    ######################################
    # Streamlit
    ######################################
    st.title("Parking Spot Finder")

    # Render the readme as markdown using st.markdown as default
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

     # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Preprocessed demo data", "Live data", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('Next, try selecting "Preprocessed demo data".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Preprocessed demo data":
        # Add horizontal line to sidebar
        st.sidebar.markdown("___")
        readme_text.empty()
        demo_mode(DEMO_VIDEO)
    elif app_mode == "Live data":
        # Add horizontal line to sidebar
        st.sidebar.markdown("___")
        readme_text.empty()
        live_mode()


    return None

def live_mode():
    # streamlit placeholder for image/video
    image_placeholder = st.empty()

    # url for video
    # Jackson hole town square, live stream
    video_url = "https://youtu.be/DoUOrTJbIu4"

    # Check for spots on temp file
    msg = """If selected, the algorithm will try to identify parking spots
            based on location of cars that don't move in the video clip.
            This works best if all parking spots are full in the first and last frames"""
    force_new_boxes = st.sidebar.checkbox("Remake parking spot map", help=msg)

    free_space_frame_cut_off = st.sidebar.slider("Count spots if open for this many frames:",
        0, 10, 0, key="spots",
        help="Parking spaces will be conisdered open if they are free for this long to avoid transiant issues")
    if st.sidebar.button("Process video clip"):
        process_video_clip(video_url=video_url,
                            image_placeholder=image_placeholder,
                            force_new_boxes=force_new_boxes,
                            free_space_frame_cut_off=free_space_frame_cut_off)


def demo_mode(DEMO_VIDEO):
    st.sidebar.write("Display a specific frame from preprocessed data or watch the entire video")
    # streamlit placeholder for image/video
    image_placeholder = st.empty()

    # Temp file to store latest clip in, should delete these later.
    total_frames = frame_count(DEMO_VIDEO, manual=True) - 1

    frame_index = st.sidebar.slider(label="Show frame:", min_value=1, max_value=total_frames, value=1,
                            step=1, key="savedClipFrame", help="Choose frame to view")

    file = r"https://github.com/rejexx/Parkingspot_Vacancy/blob/main/src/streamlit_app/demo_vacant_spots_data.pkl?raw=true"
    vacancy_per_frame = load_pickle(file)
    chart_placeholder = st.sidebar.empty()

    #Show play video button and do actions according to user input
    if st.sidebar.button("Play video"):
        display_video(image_placeholder, DEMO_VIDEO, show_chart = chart_placeholder, vacancy_per_frame = vacancy_per_frame)
    else:
        bar_chart_vacancy(vacancy_per_frame, frame_index, chart_placeholder)

        # Load the video file we want to display
        frame = display_single_frame(DEMO_VIDEO, frame_index-1)  # 0 indexed
        image_placeholder.image(frame, channels="BGR")

    #Descriptive text under the video
    st.write("Red boxes = occupied spot")
    st.write("Green boxes = available spot")
    st.markdown("This file was processed using Mask R-CNN to detect when cars overlapped with parking spots")
    st.write("Parking spaces were also identified using Mask R-CNN -  by detecting spots that had cars that didn't move")



# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/rejexx/Parkingspot_Vacancy/main/src/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


    # Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
        NAME = "coco_pretrained_model_config"
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
        DETECTION_MIN_CONFIDENCE = 0.6

@st.cache(suppress_st_warning=True)
def get_weights():     
    """Uses existing or downloads weight file from google drive"""
    save_dest = Path('coco_weights')
    save_dest.mkdir(exist_ok=True)

    #temporary location
    cloud_model_location = "1DR2t63XpubkmIhw7h75sfrqLo4DzJXul"

    f_checkpoint = Path("coco_weights/mask_rcnn_coco.h5")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model..."):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint,
            show_progress_bar = True)
    return f_checkpoint

# Make a MaskRCNN model, would be nice to cache this
def maskRCNN_model():
    """Makes a Mask R-CNN model, ideally save to cache for speed"""
    weights = get_weights()
    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir="model", config=MaskRCNNConfig())
    
    # Load pre-trained model
    model.load_weights(weights, by_name=True)
    model.keras_model._make_predict_function()
    
    return model



def process_video_clip(video_url, image_placeholder, force_new_boxes=False,
                        free_space_frame_cut_off=0):
    """Gets a video clip, uses stored parkingspot boundaries OR makes new ones,
        counts how many spots exist in each frame, then displays a graph about it
        force_new_boxes: will force creation of new parking spot boundary boxes
        video_url: YouTube video URL"""

    # Give message while loading weights
    weight_warning = st.warning("Loading model, sit tight a minute...")

    #Create model with saved weights
    model = maskRCNN_model()

    weight_warning.empty()  # Make the warning go away, done loading

    video_warning = st.warning("Getting a clip from youTube...")

    # Use pafy to the urls for video clip
    video = pafy.new(video_url)
    #get the 360p url
    medVid = video.streams[2]
    #  load a list of current segments for live stream
    playlist = m3u8.load(medVid.url)
    #get just the first clip (usually 0-7 available)
    single_segment_url = playlist.segments[0].uri
    video_warning.empty() # done loading clip
    

    parked_car_boxes = get_bounding_boxes(model, single_segment_url, force_new_boxes)

    # Process video, outputs a image to streamlit
    # This is where the real work actually happens
    count_spots_warning = st.warning("Counting spots in video")
    # def countSpots(url, parked_car_boxes, model, utils, image_placeholder,
    # video_save_file="annotatedVideo.avi",
    # framesToProcess=True, freeSpaceFrameCutOff=5, showVideo = True, skipNFrames = 10,
    # n_frames_per_segment=100, n_segments=1)
    vacancy_per_frame, image_array = countSpots(url=video_url,
                                                parked_car_boxes=parked_car_boxes,
                                                model=model,
                                                image_placeholder=image_placeholder,
                                                free_space_frame_cut_off=free_space_frame_cut_off,
                                                skip_n_frames=10)

    count_spots_warning.empty()  # Clear the warning/loading message

    #Show chart of output frames, later add some animation/rewatching ability
    bar_chart_vacancy(vacancy_per_frame)

    return None


def bar_chart_vacancy(vacancy_per_frame, frame_index=False, chart_placeholder = None):
    """Show a bar chart of vacancy per frame, with a line at 
    frame index position (if argument included)
    chart_placeholder: st. empty object for where chart should appear"""

    vacancy_per_frame_df = pd.DataFrame(
                            vacancy_per_frame, index=["Available spots"]).T
    vacancy_per_frame_df.index.name = "Frame number"
    # Altair charts can't read index, so move it to a column:
    vacancy_per_frame_df = vacancy_per_frame_df.reset_index()

    # Draw an altair chart in with information on the frame.
    chart = alt.Chart(vacancy_per_frame_df, height=220).mark_area().encode(
        alt.X("Frame number:Q", scale=alt.Scale(nice=False)),
        alt.Y("Available spots:Q")
        )

    #Add vertical line to show frame in context
    selected_frame_df = pd.DataFrame({"selected_frame": [frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")

    #display chart at given location, updating the passed chart object
    if chart_placeholder == None:
        chart_placeholder = st.empty()

    chart_placeholder.altair_chart(alt.layer(chart, vline), use_container_width=True)

@st.cache()
def load_pickle(url):
    """Loads file from pickle url"""
    # Tip from Jqadrad
    # https://stackoverflow.com/questions/61786481/why-cant-i-read-a-joblib-file-from-my-github-repo
    file = BytesIO(requests.get(url).content)
    return pickle.load(file)


# won't call the st.write if cached
@st.cache(suppress_st_warning=True)
def get_bounding_boxes(model, url, force_new_boxes=False):
    """returns bounding box as np array
    inputs: model - mask RCNN model
            url: address of video to get boxes from (if none stored)
            force_new_boxes - forces processing of video clip,
                              instead of loading parking spots from file"""
    # load or create bounding boxes
    bounding_box_file = r"https://raw.githubusercontent.com/rejexx/Parkingspot_Vacancy/main/src/streamlit_app/demo_parked_car_spots.csv"

    # Load boxes from file if they exist
    # Else process the saved file and make boxes from cars that don't move.
    if force_new_boxes == False:
        # if False:  #Force re-run
        parked_car_boxes = np.loadtxt(
            bounding_box_file, dtype='int', delimiter=',')
        st.write(
            f"Using premade parking spot map from file with {len(parked_car_boxes)} spots")
    else:
        # Learn where boxes are from movie, and save video with annotations
        # Sources is either VDIEO_SOURCE or try with tempFile
        compute_boxes_warning = st.warning(
            "No saved bounding boxes - will process to make new ones")
        # detectSpots(video_file, video_save_file, model, utils, initial_check_frame_cutoff=10):
        total_frames = frame_count(url, manual=True)
        parked_car_boxes = detectSpots(
            url, model=model, utils=mrcnn.utils, initial_check_frame_cutoff=(total_frames-5))

        # One of those 'spots' is actually a car on the road, I'm going to remove it manually
        #bad_spot = np.where(parked_car_boxes == [303,   0, 355,  37])
        #parked_car_boxes = np.delete(
        #    parked_car_boxes, bad_spot[0][0], axis=0)

        # Save edited boxes to file for future use
        #np.savetxt(bounding_box_file, parked_car_boxes, delimiter=',')
        compute_boxes_warning.empty()
        #Used for running locally, and for downloads!
        #st.write(f"Saved new bounding boxes to: {bounding_box_file}")

    return parked_car_boxes


def display_video(image_placeholder, video_file, show_chart=False, vacancy_per_frame=None):
    """Shows a video in given streamlit placeholder image
    image_placeholder: an st.empty streamlit object
    video_file: string path to video, entire video will be shown
    show_chart: st.empty object to be a chart
    vacancy_per_frame: dictionary of frame_num: count_vacant_spots"""

    # Load the video file we want to display
    video_capture = cv2.VideoCapture(video_file)
    frame_index = 0

    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        frame_index += 1
        if show_chart != False:
            bar_chart_vacancy(vacancy_per_frame, frame_index, chart_placeholder=show_chart)

        image_placeholder.image(frame, channels="BGR")
        time.sleep(0.5)

    # Clean up everything when finished
    video_capture.release()

# @st.cache(show_spinner=False)
def display_single_frame(video_file, frame_index=0):
    """Displays a single frame in streamlit
        Inputs:
        video_file - path to file name or other openCV video
        image_placeholder - streamlit st.empty() or image object
        frame_index - frame number to show, 0 indexed. Do not exceed max frames
        """
    video_capture = cv2.VideoCapture(video_file)

    video_capture.set(1, frame_index)
    success, frame = video_capture.read()

    video_capture.release()
    return frame


def get_and_process_video(url, image_placeholder,
                          n_frames_per_segment=100,
                          n_segments=7):
    '''gets frames and processes them
    returns array of images.

    Youtube segments don't cleanly exit from openCV, 
    so don't go more than 110ish frames per segment
    url: youtube URL to pull segment from
    image_placeholder: streamlit image or st.empty() obj to display video in
    n_frames_per_segment: how many frames to grab per each video clip
    n_segments: how many segments to use, each is about 5 seconds. 
        Usually only 7 exist at once.
    '''

    # Use pafy to get the 360p url
    video = pafy.new(url)

    # best = video.getbest(preftype="mp4")  #  Get best resolution stream available
    medVid = video.streams[2]

    #  load a list of current segments for live stream
    playlist = m3u8.load(medVid.url)

    # will hold all frames at the end
    # can be memory intestive, so be careful here
    frame_array = []

    #  Clip to total size if key word used
    if n_segments == "all":
        n_segments = int(len(playlist.segments))

    #  Loop through all segments
    for i in playlist.segments[0:7]:

        capture = cv2.VideoCapture(i.uri)

        #  go through every frame in segment
        for i in range(n_frames_per_segment):
            success, frame = capture.read()  # read in single frame

            if success == False:
                break

            image_placeholder.image(frame, channels="BGR")
            i += 1

            frame_array.append(frame)

        capture.release()

    return frame_array


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


def write_frames_to_file(frame_array, file_name="video.avi", nth_frames=1, fps=15):
    '''writeFramesToFile(frame_array, fileName="video.avi", nthFrames=1, fps=15)
    Writes array of images to a video file of type .avi
    parameters:
      frame_array - python list of frames as pixel values from openCV
      fileName - path to save file
      nthFrames - how many frames to keep, 
        1 will keep all frames, 2 will remove every other, etc...
      fps - frames per second'''
    assert (len(frame_array) > 0), "No images to save, frame_array is empty"

    # check first frame and find shape
    width = len(frame_array[0][0])
    height = len(frame_array[0])
    size = (width, height)

    # make video writer
    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(
        *'DIVX'), fps, size)  # name, writer, fps, size

    for i in range(0, len(frame_array), nth_frames):
        out.write(frame_array[i])
    out.release()
    return None


@st.cache()
def frame_count(video_path, manual=False):
    """frame_count - get how many frames are in a video
    video_path - path to video
    manual - True or False, manual method is much more accurate but slow"""
    # Credit: https://stackoverflow.com/questions/25359288/
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


def countSpots(url, parked_car_boxes, model, image_placeholder,
               frames_to_process=True, free_space_frame_cut_off=5, show_video=True, skip_n_frames=10,
               n_frames_per_segment=100, n_segments=1):
    '''Counts how many spots are vacant at the end of the video 
    saves a video showing spots being vacant
    returns: count of spots in final frame that are vacant 
             AND array of processed frames as a tuple, be sure to unpack
    inputs: url - youtube url
            parked_car_boxes: bounding boxes of parking spaces
            model: Mask R-CNN inference model object
            image_placeholder: streamlit st.empty or image object
            video_save_file: default "annotatedVideo.avi", video with annotations
            framesToProcess: default = all frames (True).  
                             Enter an int to process only part of the file. Good for saving time
            freeSpaceFrameCutOff: default 2, number of frames a spot must be empty before appearing
                                  as such, helps with jitter
            n_frames_per_segment: integer, how many frames to display (>110 is not stable)
            n_segments: number of video segments to show, typically there are 7
                        will trim to max if you exceed it or type 'all'''

    assert (skip_n_frames >
            0), "skipNFrames must be greater than 0. Default is 1 (no skipping)"
    assert (free_space_frame_cut_off >=
            0), "freeSpaceFrameCutOff can't be negative. Default is 5"

    # Use pafy to get the 360p url
    video = pafy.new(url)

    # best = video.getbest(preftype="mp4")  #  Get best resolution stream available
    medVid = video.streams[2]

    #  load a list of current segments for live stream
    playlist = m3u8.load(medVid.url)

    # will hold all frames at the end
    # can be memory intestive, so be careful here
    frame_array = []

    # Speed processing by skipping n frames, so we need to keep track
    frame_num = 0

    # How many free spots per frame, #frame_Num:vacant spots
    vacancy_per_frame = {}

    #  Clip to total size if key word used
    if n_segments == "all":
        n_segments = int(len(playlist.segments))

    # Dictionary of parking space index and how many frames it's been 'free'
    car_boxes_open_frames = {i: 0 for i in range(len(parked_car_boxes))}

    # Loop over each frame of video
    #  Loop through all segments
    for i in playlist.segments[0:7]:

        capture = cv2.VideoCapture(i.uri)

        # go through every frame in segment
        for i in range(n_frames_per_segment):

            success, frame = capture.read()
            if not success:
                break

            # Skip every nth frame to speed processing up
            if (frame_num % skip_n_frames != 0):
                frame_num += 1
                pass
            else:
                frame_num += 1

                # Convert the image from BGR color (which OpenCV uses) to RGB color
                rgb_image = frame[:, :, ::-1]

                print(f"Processing frame: #{frame_num}")
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

                if len(car_boxes) == 0:
                    #If there are no cars spotted, set all overlaps to 0
                    overlaps = [0.0]*len(parked_car_boxes)
                else:
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
                        # If the spot has appeared open long enough, count it as free!
                        # This is so we don't alert based on one frame of a spot being open/closed.
                        # This helps prevent the script triggered on one bad detection.
                        if car_boxes_open_frames[row]+1 >= free_space_frame_cut_off:
                            # Parking space not occupied! Draw a green box around it
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 1)

                            # Count this as occupied, and don't let it be immediately changed
                            car_boxes_open_frames[row] = max(
                                free_space_frame_cut_off, 1)
                        else:
                            # Parking space hasn't been vacant long enough - draw a red box around it
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 0, 255), 1)

                            # Tag this spot as being open for +1 frame
                            car_boxes_open_frames[row] += 1

                    else:
                        # else, spot appears occupied this frame
                        # If it's been occupied for more than the frame cutoff:
                        if car_boxes_open_frames[row] <= free_space_frame_cut_off:
                            # Parking space is occupied - draw a red box around it
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 0, 255), 1)
                            # Set the frame counter to 0, it's full, don't let it change immediately
                            car_boxes_open_frames[row] = 0
                        else:
                            # Start counting frames this spot is full,
                            # So script isn't triggered from someone driving by
                            car_boxes_open_frames[row] -= 1

                            # Parking space still 'free'. Draw a green box around it
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 1)

                    # Write the IoU measurement inside the box
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f"{max_IoU_overlap:0.2}",
                                (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

                # If a space has been free for several frames, let's count it as free
                #  loop through all 'free' frames and sum the result
                free_spaces = sum(
                    [int(i) > free_space_frame_cut_off for i in car_boxes_open_frames.values()])

                # Save num free spaces in frame to dict for final output
                vacancy_per_frame[frame_num] = free_spaces

                # Write how many free spots there are at the top of the screen
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    frame, f"Free Spaces: {free_spaces}", (30, 30), font, 1.0, (0, 255, 0), 2, cv2.FILLED)

                print(f'Free Spaces: {free_spaces}')
                # print number of frames
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame, f"Frame: {frame_num}",
                            (10, 340), font, 0.5, (255, 255, 255))

                # Show the video placeholder
                if show_video:
                    image_placeholder.image(frame, channels="BGR")
                    time.sleep(0.01)

                # Append frame to outputvideo
                frame_array.append(frame)

                if (frames_to_process != True and frame_num > frames_to_process):
                    print(
                        f"Stopped processing at frame {frame_num} as requested by framesToProcess parameter")
                    break

    # Clean up everything when finished
    capture.release()  # free the video
    # writeFramesToFile(frame_array=frame_array, fileName=video_save_file) #save the file

    return (vacancy_per_frame, frame_array)


def detectSpots(video_file, model, utils,
                 show_video=True, initial_check_frame_cutoff=10):
    '''detectSpots(video_file, initial_check_frame_cutoff=10)
    Returns: np 2D array of bounding boxes of all bounding boxes that are still occupied
    after initial_check_frame_cutoff frames.  These can be considered "parking spaces".

    An update might identify any spaces that get occupied at some point and stay occupied 
    for a set length of time, in case some areas start off vacant.'''
    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(video_file)

    # Store the annotated frames for output to video/counting how many frames we've seen
    frame_array = []

    # Will contain bounding boxes of parked cars to identify 'parkable spots'
    parked_car_boxes = []
    parked_car_boxes_updated = []

    # Make image appear in streamlit
    image_placeholder_processing = st.empty()

    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            st.write("Processed {len(frame_array)} frames of video, exiting.")
            return parked_car_boxes

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = frame[:, :, ::-1]

        # ignore the inbetween frames 0 to x, don't run the model on them and save processing time
        if 0 < len(frame_array) < initial_check_frame_cutoff:
            print(f"ignore this frame for processing, #{len(frame_array)}")
        else:
            print(f"Processing frame: #{len(frame_array)}")
            # Run the image through the Mask R-CNN model to get results.
            # model.keras_model._make_predict_function() #Will this solve my bug?
            # Or try this next https://stackoverflow.com/questions/54652536/keras-tensorflow-backend-error-tensor-input-10-specified-in-either-feed-de

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
                parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
                parked_car_boxes_init = parked_car_boxes
                print('Parking spots 1st frame:', len(parked_car_boxes))

            # If we are past the xth initial frame, already know where parked cars are, then check if any cars moved:
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
                        # In the first few frames, remove this 'spot' and consider it as a moving car instead
                        # Transient event, draw green box
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 3)
                    else:
                        # if len(frame_array) == initial_check_frame_cutoff:
                        # Consider this a parking spot, car is still in it!
                        # Dangerous to mutate array while using it! So make a new one
                        parked_car_boxes_updated.append(list(parking_area))

                        # Parking space is still occupied - draw a red box around it
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 0, 255), 1)

                    # Write the top and bottom corner locations in the box for ref
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, str(parking_area),
                                (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

                parked_car_boxes = np.array(
                    parked_car_boxes_updated)  # only happens once

        # print number of frames
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, f"Frame: {len(frame_array)}",
                    (10, 340), font, 0.5, (0, 255, 0), 2, cv2.FILLED)

        # Show the frame of video on the screen
        if show_video:
            image_placeholder_processing.image(frame, channels="BGR")
            time.sleep(0.01)

        # Append frame to outputvideo
        frame_array.append(frame)

        # stop when cutoff reached
        if len(frame_array) > initial_check_frame_cutoff:
            print(f"Finished, processed frames: 0 - {len(frame_array)}")
            break

    # Clean up everything when finished
    video_capture.release()
    #write_frames_to_file(frame_array=frame_array, file_name=video_save_file)

    # Show final image in matplotlib for ref
    return parked_car_boxes


if __name__ == "__main__":
    main()
