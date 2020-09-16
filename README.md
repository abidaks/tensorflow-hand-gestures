Tensorflow Hand Gestures using Posenet
====================================

This project is based on tensorflow pose estimation
Mediapipe provides hand gestures for all other platforms but not for python.
So in this repo you can get hand gestures for python


## About this module

It can track hand movement for both left and righ thand seperately

Left Hand:
Left to right movement
Right to left movement
Swipe Up movement
Swipe Down Movement

Right Hand:
Left to right movement (Swipe right)
Right to left movement (Swipe left)
Swipe Up movement
Swipe Down Movement


You can use this gesture tracking code to do lots of tasks like next screen on swipe left or right or you can use it to play a game in rel time.


## Installing this module

> **Requirements**
>
> This project has been tested in Ubuntu 18.04 with Python 3.6.5. Further package requirements are described in the
> `requirements.txt` file.
> - It is a requirement to have [Tensorflow>=1.14.0 installed](https://www.tensorflow.org/install/pip) (either in gpu 
> or cpu mode). This is not listed in the `requirements.txt` as it [breaks GPU support](https://github.com/tensorflow/tensorflow/issues/7166). 
> - Run `python -c 'import cv2'` to check that you installed correctly the `opencv-python` package (sometimes
> [dependencies are missed](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo) in `pip` installations).

To start using this framework clone the repo:

```bash
git clone https://github.com/abidaks/tensorflow-hand-gestures
cd tensorflow-gesture-tracking
pip install -r requireents.txt
```
Once you installed the requirement, run below code to download models
```
python wget.py
```

## Running the program

when you finished installing everything, use the below command to run the code enjoy.
```
python tracker.py
```

Make sure you have camera attached to your pc or laptop.

## Future Developments
I will add waving gesture for both hands seperately in the future.

## Acknowledgements

The posenet model is created by Google and can be found [here](https://github.com/tensorflow/tfjs-models/tree/master/posenet).

