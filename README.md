Spot or Not? Parking Spot Vacancy with Machine Learning
==============================

Finding available parking spots in Jackson Hole town square using a camera and machine learning.  This is my final capstone project, part of Springboard's six month Data Science course. 

**Check out the streamlit app:** [Parking vacancy](https://share.streamlit.io/rejexx/parkingspot_vacancy/main/src/streamlit_app.py)

See final report: [Spot or Not Final report](https://github.com/rejexx/Parkingspot_Vacancy/raw/main/reports/Final%20Report%20Spot%20or%20Not.pdf)

Here's some highlights:

* Use Mask R-CNN to detect cars occupying spots
* Deployed the app to public streamlit sharing
* Pull a clip from a youtube live stream to check for spots, any time, always latest stream
* Can be easily generalized to another parking lot camera given:
    * Picture/video of parking lot when it's 100% full (to identify spots)
    * Youtube URL pull the stream from

![Jackson Hole townsquare with instance segmentation masks](https://github.com/rejexx/Parkingspot_Vacancy/raw/main/reports/figures/townsquare_with_masks.png)
_Jackson Hole town square with instance segmentation masks from Mask R-CNN. labeled on each detected instance are the classification (person, car) and confidence of classification (between zero and one)__

## Background

### Problem 

It's hard to find a parking spot in busy places (like Jackson Hole's town square).  It would be great to know how many spots were available on demand, with potential to keep some record.  Not only would this help travelers, it could be used by city planners to identify under used parking or areas that need more parking.  Everyone wants to get a selfie with the [elk antler arches](https://www.jacksonholewy.com/blog/elk-arches-town-square/), why make them wonder if town square parking is available? This is 2021 - there's data for that.

### Objective

Use webcam data to detect which spots have cars in them using machine learning!  If google can recognize faces in photographs, why not recognize if a parking spot has a car in it or not?

### Outcomes

![Images of parking lot with boxes](https://github.com/rejexx/Parkingspot_Vacancy/raw/main/reports/figures/sample_frame161_boxes.jpeg)

![Graph of parking data](https://github.com/rejexx/Parkingspot_Vacancy/raw/main/reports/figures/graph_frame161.png) _Sample of processed data, Frame 161, showing red occupied spots and green vacant spots. The number indicates how much of the spot is occupied by a car (0 = not occupied, 1= completely filled)_



To measure accuracy, I took every 10th frame of the 175 demo video clip (which covers both night and day) and counted parking spaces.  Considering myself an expert at finding available parking spaces, I used the human count as "Truth" to compute the following confusion matrix and derived metrics.

| Accuracy             | 0.89 |
| -------------------- | ---- |
| Misclassification    | 0.11 |
| Precision            | 0.57 |
| Sensitivity (recall) | 0.15 |
| Specificity          | 0.87 |

<img src="https://github.com/rejexx/Parkingspot_Vacancy/raw/main/reports/figures/confusion_matrix.png" alt="Confusion matrix  table" style="zoom:70%;" />

Take note - this model gives zero false negatives.  This means that whenever a parking spot was marked as 'taken' it was always correct!  The number of false positives indicates that the model often predicts spots as 'available' when they are not.  I saw three common reasons for misclassification.

Common misclassification causes and how to fix them:

- Parking spots that are very few pixels (in the back) were frequently misclassified.  Use higher resolution or ignore those spots.  This could bring accuracy as high as 0.96 (!).
- Black or white cars were often misclassified.  Higher resolution, or tuning the model would probably help.
- Temporary occlusion of spaces (such as a garbage truck or bus driving in front of them) caused the model to think spaces were 'available' because the space was blocked by something it didn't recognize as a car.  This can be fixed by waiting a few extra frames before declaring a spot as 'vacant' to make sure its really vacant.  This code is already implemented in the 'live demo' version.

#### Additional Use Cases

Data on how full a parking lot is can also be used for:

* Proxy for number of customers visiting a business over time
* Proxy for how busy an area is - for example: shopping malls, airports, conference centers.
* Identify under utilized portions of parking lots to flag for repurposing (food trucks anyone 🌮?)
* Identify commonly used 'parking spots' that aren't allowed (parking illegally)

### Method

I used Mask R-CNN (Region-proposal Convolutional Neural Network) [Original Mask R-CNN paper by facebook research](https://arxiv.org/abs/1703.06870) to process each video frame to identify cars and trucks.  If a car or truck's bounding box overlaps with a designated parking place, that spot is considered occupied.  If not, the spot is vacant.  I used an updated version that works with TensorFlow 2 [Repo here](https://github.com/akTwelve/Mask_RCNN).  To save time training, I used the [Matterport model weights](https://github.com/matterport/Mask_RCNN), trained on the COCOs dataset.

**Step 1:** Identify parking spots

- I took a video clip during 'rush hour' when no spots were empty, and ran Mask R-CNN to detect vehicles.  The idea being to identify any cars that didn't move, and assume they were parked.  In other words, I compared the first frame against another frame a few minutes later. Any vehicle bounding boxes from the first frame that were still full in the later frame were considered to be parking spaces.  Any boxes that weren't full were considered as noise (i.e. a car driving on the road). Big shoutout to [Adam Geitgey's article](https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400) for this idea.
- Other potential methods people have used:
  -  training a specialized neural network to identify empty (or full) spaces. See [cnrpark](http://cnrpark.it/)
  -  Looking at painted lines (Jackson Hole's town square's prime spots are parallel parking that don't have any lines). See  [Dwivedi's article](https://towardsdatascience.com/find-where-to-park-in-real-time-using-opencv-and-tensorflow-4307a4c3da03).
  - Have somebody draw boxes manually, or [map it with a drone](https://medium.com/geoai/parking-lot-vehicle-detection-using-deep-learning-49597917bc4a)
  
    

**Step 2:** Determine if spots are full

* I compare the parking spot bounding box with all the cars bounding boxes. If there's a significant overlap, the spot is considered full.
* This is done using IoU - intercept on union.  IoU compares how much of the car's box overlaps with the parking spot box.  If it's above a threshold, I count the spot as being occupied.  Side note: IoU is used internally by R-CNN networks to give one bounding box per instance and avoid double-counting objects.  This means there's already great functions to compare if two boxes overlap significantly or not.  

## Challenges

* Most parking lot cameras are not situated at a very good angle for detecting available spots.  They are only ~10 feet off the ground and the first line of cars obscures all the other parking lots.  The higher up and the more 'in-line', the easier it is for the algorithm to tell if spots are there.  The Jackson Hole camera offers several different types of parking, so it's a decent test.
* It took 10% of my time getting the image detection model up and running on my laptop, 40% getting code to reliably and smoothly pull youTube clips and 50% getting it working in streamlit and deployed streamlit sharing.  On the bright side, streamlit's servers are much faster than my little laptop 🏎️.


## Future Improvements

**Algorithm and statistics**
* Try using a higher resolution stream to capture cars farther from the camera, and add in blurring faces of pedestrians (for privacy).
* Try YOLO algorithm, which is faster but less accurate than R-CNN.  Since I'm not currently using the mask, there's no good reason to spend extra processing time obtaining it
* More samples on model accuracy vs. human counting open/filled spots
* Try [cnrpark's](http://cnrpark.it/) training images for open/vacant parking spots and see if it performs better than car detection with IOU. 
* Try on other parking lots (such as Jackson Hole's airport)
* Currently struggles with black cars or trucks, try re-training the model, or using higher resolution stream to resolve this.

**App Features**
* Ability to replay processed video
* Add options to run on any video:
   * Enter your own video URL or upload one
   * Download parking place bounding boxes after processing
   * Upload saved parking place bounding boxes
   * Download processed video
* Store 'parking lot vacancy' data to glean insights about location.

## Other References

* Cazamias, Marek, 2016. Parking Space Classification using Convolutional Neural Networks.   [See paper](http://cs231n.stanford.edu/reports/2016/pdfs/280_Report.pdf).  They use a single image to count parking spots.
* Acharya, Yan. Real-time image-based parking occupancy detection using deep learning [See paper](http://ceur-ws.org/Vol-2087/paper5.pdf)
* [COCO dataset for training](http://cocodataset.org/)

### Where is this camera from?

![Elk antler arch](https://jhcr.s3.amazonaws.com/3323/arch-summer__large.jpg) 
__The famous [elk antler arch](https://www.jacksonholewy.com/blog/elk-arches-town-square/) seen in the video__
* Learn more about [Jackson Hole cameras](https://www.seejh.com/webcams/jackson-hole-featured-webcams/town-square-southwest/)
* I got permission from Robert Strobel, seeJH.com CEO to use this camera for this purpose.  If you want to use my code on other cameras, keep in mind [YouTubes policy about face detection](https://www.youtube.com/t/terms). "You are not allowed to (...) collect or harvest any information that might identify a person (for example, usernames or faces), unless permitted by that person or allowed under section (3) above;"  I specifically used a low resolution (360p) version of this camera to make it hard to identify faces.

Project Organization 
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, processed video files and bouding boxes
    │   └── raw            <- The original video files
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── packages.txt       <- The packages for reproducing the analysis environment, also used by streamlit sharing
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for streamlit app
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── clip_grabber.py
    │   │
    │   ├── mrcnn           <- Mask R-CNN library from [this fork of Matterport]((https://github.com/akTwelve/Mask_RCNN)
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── streamlit_app     <- Assets for streamlit app
    │   ├── streamlit_app.py  <- Streamlit app
    │   |
    |   ├── requirements.txt  <- The requirements file for reproducing the analysis environment, e.g.
    │   |                      generated with `pip freeze > requirements.txt`
    │   │                     
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
(Not all folders are currently being used as described, leaving room for growth)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
