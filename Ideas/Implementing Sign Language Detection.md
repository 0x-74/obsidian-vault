there are two main ways to approach sign language detection:
- end to end deep learning models 
	-  this approach is very low in complexity but requires a lot of training data,it directly tries to maps the input images to the outputs
- using a multi step process with landmarks for detecting 
	-  this approach consists of 3 steps :
		- detecting the hands and creating bounding boxes 
		- using a landmark creation model to draw the landmarks and retrieve the coordinates of them 
		- using a small neural network to learn the mappings of landmark coordinates to the output labels
## current approach for multi step classification
- using a mediapipeline model to detect the hands and build bounding boxes around them 
- using another mediapipeline model to draw landmarks 
- inputting all the videos of the gestures and treating each video as an label
- training the model
- testing and reiterating till achieving required accuracy
## results
- the current approach fails on real time gesture detection as a gesture can have multiple movements.
- there is also ambiguity in when a gesture ends and starts
## second pass
- levenshtein distance evaluation rather than accuracy
- weighted classes to kill ambiguity of similiar gestures
- sliding window for temporal gesture
- temporal segment networks for segmenting video into segments