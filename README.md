## Face recongition ( Model and Flask app )

### Model 
The models fih how to detect/recognitze faces in both images and videos.

1. To test image:
go to directory and run : 
 python3 recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle --image images/mohamed.jpg

2. To test video: 

Go to directory and run:

 python3 recognize_video.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle




### Flask Video [ Almost done with it ]

### Usage 

Go to Directory and run: 


python3 main.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
