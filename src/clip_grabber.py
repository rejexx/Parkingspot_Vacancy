# IMPORTS
# Video getting and saving
import cv2  # open cvs, image processing
import m3u8
import pafy  # needs youtube_dl
import time

def get_frames_from_video(url, n_frames_per_segment=1):
    '''gets frames and processes them
    returns array of images.

    Youtube segments don't cleanly exit from openCV, 
    so I am instead chopping them off a little bit
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

    segment = playlist.segments[0]

    capture = cv2.VideoCapture(segment.uri)

    #  go through n frame in segment
    for segment in range(n_frames_per_segment):
        success, frame = capture.read()  # read in single frame

        if success == False:
            break

        frame_array = frame

    capture.release()

    return frame_array

def append_frames_to_file(frame_array, file_name="video.mpeg", nth_frames=1, fps=15):
    '''writeFramesToFile(frame_array, fileName="video.avi", nthFrames=1, fps=15)
    Writes array of images to a video file of type .avi
    parameters:
      frame_array - python list of frames as pixel values from openCV
      fileName - path to save file
      nthFrames - how many frames to keep, 
        1 will keep all frames, 2 will remove every other, etc...
      fps - frames per second'''
    assert (len(frame_array) > 0), "No images to save, frame_array is empty"

    #if file_name doesn't exist, we need to make it

    # check first frame and find shape
    width = len(frame_array[0][0])
    height = len(frame_array[0])
    size = (width, height)

    capture = cv2.VideoCapture(file_name)

    all_frames = []
    #Get old video and put in memory
    # each time you do this the video gets worse
    #   Like a bad photocopier.  Try to avoid re-writing many times
    #   After 10+ writes, the video becomes terrible quality
    #   I think this is due to bad compression
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        all_frames.append(frame)

    capture.release()

    # Append all frames into one complete whole
    [all_frames.append(frame) for frame in frame_array]

    # make video writer, write frames to file
    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(
        *'DIVX'), fps, size)  # name, writer, fps, size

    for i in range(0, len(all_frames), nth_frames):
        out.write(all_frames[i])
    out.release()
    return None   


url = "https://youtu.be/DoUOrTJbIu4"
save = "C:\\springboard\\Parkingspot_Vacancy\\data\\raw\\all_day4.avi"
wait_time = 30 # seconds
total_iterations = 200
frame_array = []

for i in range(total_iterations):
    # get one frame (default)
    frame = get_frames_from_video(url)
    # append it to file
    frame_array.append(frame)

    print(f"Got clip {i+1}/{total_iterations}")
    time.sleep(wait_time)

#openCVProcessing(frame_array)

append_frames_to_file(frame_array, file_name=save, fps=3)
