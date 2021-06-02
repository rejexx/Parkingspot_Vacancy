Parkingspot_Vacancy
==============================

Finding available parking spots using a camera and machine learning.  This is my final capstone project for Springboard Data Science bootcamp course. 

**Check out the streamlit app:** [Parking vacancy](https://share.streamlit.io/rejexx/parkingspot_vacancy/main/src/streamlit_app.py)

I'm finishing up this project mid June and will add some nicer documentation by that point!  Here's some highlights:

* Use Mask R-CNN to detect cars occupying spots
* Deploy the app to streamlit 
* Pull a clip from a youtube live stream to check for spots, any time, unique data
* Can be easily generalized to another parking lot camera given:
    * picture/video of parking lot when it's 100% full (to identify spots)
    * Youtube URL to get a new clip from

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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
