# Machine-Learning-Model-Training

# This is Yolov3 training kit. I have train my own model based on darknet method using yolov3-tiny and yolov3 only.
This model is based on my Hand Gesture dataset.

Train model step:
a. Login into google colab and mount jupyterlab it into google drive
b. In google drive create following folder:
   1. Create Main Folder called as - yolov3-tiny_custom_model_training
   2. Inside Main Folder create subdirectories as - backup, custom data, custom_weight, darknet
   3. Once done, go to current PC and create our own dataset at least 300 images jpg format. In my case, i'm using my hand gesture dataset that was captured using capture_photo_webcam.py script
   4. Next, we have to create labelling and annotations for our image. The easies way is to use https://www.makesense.ai/ (Upload the photos and label each photos and download it as Yolo Format)
   5. Go into custom_data directory and then copy the *.txt from downloaded files just now into our custom_data directory which should consist of "images + *.txt"
   6. Inside this directory create a file named as "classess.name" that should consist dataset labelling names for example "Right", "Left", "Up", "Down", "Stop"
   7. Inside this directory copy a script "creating-files-data-and-name.py" and "creating-train-and-test-txt-files.py". Edit this file and set "full_path_to_images= 'custom_myhand_dataset'"
   8. Then clone this file into our darknet directory git clone 'https://github.com/AlexeyAB/darknet.git' '/content/drive/MyDrive/yolov3-tiny_custom_model_training/darknet/'
   9. Once completed, go into darknet directory and open "Makefile". Apply setting as follow:
      GPU=1
      CUDNN=1
      CUDNN_HALF=0
      OPENCV=1
      AVX=0
      OPENMP=0
      LIBSO=0
      ZED_CAMERA=0
      ZED_CAMERA_v2_8=0
  10. Then go into cfg and edit yolov3-tiny.cfg (We'training yolov3-tiny pre-trained model)
      For Training we have to uncomment
      Training
      batch=64
      subdivisions=16
      
      max_batches = number of classes * 2000 (3 * 2000)
      steps = must be 20% than max_batches (5800, 6200)
      
      filters = use this equation (classes + 5) * 3 ---> (3 + 5) * 3 = 24
      [yolo]
      classess = number of classes (3 for example)
      
       save and exit
  11. Copy pre-trained model (yolov3-tiny.conv.11) into custom_weight directory
      
  12. cd into /darknet/ directory and run !make (To run the compilations)
  13. once complete then cd into /yolob3-tiny_custom_model_training and run the scripts inside custom_data
      !python custom_data/creating-files-data-and-name.py
      !python custom_data/creating-train-and-test-txt-files.py
      
  14. Then run the training as follow :
      !darknet/darknet detector train custom_data/labelled_data.data darknet/cfg/yolov3-tiny.cfg custom_weight/yolov3-tiny.conv.11 -dont_show
  15. Wait for about 4~5 hours of training
  16. Once completed, edit and comment yolov3-tiny.cfg Training
      #Training
      #batch=64
      #subdivisions=16
      
      Uncomment to the the model
      #Testing
      batch=1
      subdivisions=1
      
  17. To test our model, can use tensor_yolov3_custom_object_detection_weight.py script
