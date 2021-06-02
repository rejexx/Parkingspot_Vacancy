# App Instructions

Use the left sidebar selector to choose what to do:

1. See a demo with preprocessed data (runs quick, good place to start)

2. Get live data from the YouTube stream and find parking spots

3. View the source code (you can link directly to the gitHub repo via the upper right menu)

   

## Background

### Problem

It's hard to find a parking spot in busy places (like Jackson Hole's town square).  It would be great to know how many spots were available at any given time, and keep some record  Not only would this help travelers, it could be used by city planners to identify under used parking or areas that need more parking.  

### Solution

Use webcam data to detect which spots have cars in them using machine learning!  If google can recognize faces in photographs, why not recognize if a parking spot has a car in it or not?

### Method

I used Mask R-CNN (Region-proposal Convolutional Neural Network) [Original Mask R-CNN paper](https://arxiv.org/abs/1703.06870) to process each video frame to identify cars and trucks.  If a car or truck's bounding box overlaps with a designated parking place, we'll call that spot occupied.  If not, the spot is vacant.  I used an updated version that works with TensorFlow 2 [Repo here] (https://github.com/akTwelve/Mask_RCNN).  To save time training, I used the [matterport model weights](https://github.com/matterport/Mask_RCNN), trained on the COCOs dataset.

**Step 1:** Identify parking spots

- Method used here: I took a video clip during 'rush hour' when no spots were empty, and ran Mask R-CNN to detect vehicles.  The hope was to identify any cars that didn't move, and assume they were parked.  In other words, I compared the first frame against another frame a few minutes later. Any vehicle bounding boxes from the first frame that were still full in the later frame were considered to be parking.  Big shoutout to [Adam Geitgey's article] (https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400) for this idea.
- Other potential methods
  -  training a specialized neural network to identify empty (or full) spaces. See [cnrpark](http://cnrpark.it/)
  -  Looking at painted lines (Jackson Hole's town square's prime spots are parallel parking that don't have any lines). See  (Dwivedi's article)[https://towardsdatascience.com/find-where-to-park-in-real-time-using-opencv-and-tensorflow-4307a4c3da03]
  - Have somebody draw boxes manually, or [map it with a drone] (https://medium.com/geoai/parking-lot-vehicle-detection-using-deep-learning-49597917bc4a)
  - 

**Step 2:** Determine if spots are full

* I compare the parking spot bounding box with all the cars bounding boxes. If there's a significant overlap, we'll say the spot is full.
* This is done using IoU - intercept on union.  IoU compares how much of the car's box overlaps with the parking spot box.  If it's above a threshold, I count the spot as being occupied.  Side note: IoU is used internally by R-CNN networks to give one bounding box per instance and avoid double-counting objects.  This means there's already great functions to compare if two boxes overlap significantly or not.  

There's options for extra processing, like how many frames a spot needs to appear vacant or occupied in a row before counting it. This helps 'smooth' out the signal from transient events, like a car driving by obscuring the parking spot.

## Other References

* Cazamias, Marek, 2016. Parking Space Classification using Convolutional Neural Networks.   [See paper](http://cs231n.stanford.edu/reports/2016/pdfs/280_Report.pdf).  They use a single image to count parking spots.
* Acharya, Yan. Real-time image-based parking occupancy detection using deep learning [See paper](http://ceur-ws.org/Vol-2087/paper5.pdf)
* [COCO dataset for training](http://cocodataset.org/)