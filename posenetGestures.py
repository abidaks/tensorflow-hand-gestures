#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import cv2
import yaml
import time
import json
import math
import struct
import numpy as np
import tensorflow.compat.v1 as tf

from decode_multi_pose import decodeMultiplePoses
from draw import drawKeypoints, drawSkeleton, drawPositions

from gestureTracker import GestureTracker

tf.disable_v2_behavior()

color_table = [(0,255,0), (255,0,0), (0,0,255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

isUser = False

userName = ""

soundStartTime = time.time()
firstSoundPlay = True
helloUser = False
helloUserStartTime = time.time()
moving_commands = ["walk" , "dance" , "jump"]
soundsDurations = {"hello" : 2 , "notUser" : 3 , "onePerson" : 4 , "again" : 3}

class posenetGesture:
    def __init__(self):
        self.gestures = GestureTracker()
        with open('config.yaml') as f:
            cfg = yaml.load(f)
            checkpoints = cfg['checkpoints']
            imageSize = cfg['imageSize']
            chk = cfg['chk']
            self.outputStride = cfg['outputStride']
            chkpoint = checkpoints[chk]
            self.mobileNetArchitectures = self.architecture(chkpoint, cfg)
            self.width = imageSize
            self.height = imageSize
            self.layers = self.toOutputStridedLayers()
            self.variables(chkpoint)
            
            self.wav_obj = None
            self.tobeplayed = []
            self.waitTime = 2;
            self.waitTimeCorordin = 2;
            self.startAvatar = True

    def variables(self, chkpoint):
        with open(os.path.join('./waits/', chkpoint, "manifest.json")) as f:
            self.variables = json.load(f)
            # with tf.variable_scope(None, 'MobilenetV1'):
            for x in self.variables:
                filename = self.variables[x]["filename"]
                with open(os.path.join('./waits/', chkpoint, filename), 'rb') as fp:
                    byte = fp.read()
                    fmt = str(int(len(byte) / struct.calcsize('f'))) + 'f'
                    d = struct.unpack(fmt, byte)
                    d = tf.cast(d, tf.float32)
                    d = tf.reshape(d, self.variables[x]["shape"])
                    self.variables[x]["x"] = tf.Variable(d, name=x)
        return None
    
    def architecture(self, chkpoint, cfg):
        if chkpoint == 'mobilenet_v1_050':
            mobileNetArchitectures = cfg['mobileNet50Architecture']
        elif chkpoint == 'mobilenet_v1_075':
            mobileNetArchitectures = cfg['mobileNet75Architecture']
        else:
            mobileNetArchitectures = cfg['mobileNet100Architecture']
        return mobileNetArchitectures

    def toOutputStridedLayers(self):
        currentStride = 1
        rate = 1
        blockId = 0
        buff = []
        for _a in self.mobileNetArchitectures:
            convType = _a[0]
            stride = _a[1]
            if (currentStride == self.outputStride):
                layerStride = 1
                layerRate = rate
                rate *= stride
            else:
                layerStride = stride
                layerRate = 1
                currentStride *= stride
            buff.append({'blockId': blockId, \
                         'convType': convType, \
                         'stride': layerStride, \
                         'rate': layerRate, \
                         'outputStride': currentStride})
            blockId += 1
        return buff

    def convToOutput(self, mobileNetOutput, outputLayerName):
        w = tf.nn.conv2d(mobileNetOutput, \
                         self.weights(outputLayerName), \
                         [1,1,1,1], padding='SAME')
        w = tf.nn.bias_add(w, self.biases(outputLayerName), name=outputLayerName)
        return w

    def conv(self, inputs, stride, blockId):
        return tf.nn.relu6(
            tf.nn.conv2d(inputs, \
                         self.weights("Conv2d_" + str(blockId)), \
                         stride, padding='SAME')
            + self.biases("Conv2d_" + str(blockId)))

    def weights(self, layerName):
        return self.variables["MobilenetV1/" + layerName + "/weights"]['x']

    def biases(self, layerName):
        return self.variables["MobilenetV1/" + layerName + "/biases"]['x']

    def depthwiseWeights(self, layerName):
        return self.variables["MobilenetV1/" + layerName + "/depthwise_weights"]['x']

    def separableConv(self, inputs, stride, blockID, dilations):
        if (dilations == None):
            dilations = [1,1]
        dwLayer = "Conv2d_" + str(blockID) + "_depthwise"
        pwLayer = "Conv2d_" + str(blockID) + "_pointwise"
        w = tf.nn.depthwise_conv2d(inputs, \
                                   self.depthwiseWeights(dwLayer), \
                                   stride, 'SAME', rate=dilations, data_format='NHWC')
        w = tf.nn.bias_add(w, self.biases(dwLayer))
        w = tf.nn.relu6(w)
        w = tf.nn.conv2d(w, self.weights(pwLayer), [1,1,1,1], padding='SAME')
        w = tf.nn.bias_add(w, self.biases(pwLayer))
        w = tf.nn.relu6(w)

        return w


    
    def get_only_poses_with_valid_score(self,poses):
        """
        

        Parameters
        ----------
        poses : list
            All poses list.

        Returns
        -------
        successPoses : list
            array of pose with score is > 0.2.

        """
        successPoses = []
        for pose in poses:
            if pose["score"] > 0.2:
                successPoses.append(pose)
        return successPoses


    
    def getWeights(self):
        weights = np.ones((1,2,17))
        
        weights[0][0][0] = 0
        weights[0][1][0] = 0
        weights[0][0][7] = 0
        weights[0][1][7] = 0
        weights[0][0][9] = 0
        weights[0][1][9] = 0
    
        return weights

    def load_model(self):
        weights = self.getWeights()
        self.image = tf.placeholder(tf.float32, shape=[1, self.width, self.height, 3],name='image')
        x = self.image
        rate = [1,1]
        buff = []
        with tf.variable_scope(None, 'MobilenetV1'):
            for m in self.layers:
                strinde = [1,m['stride'],m['stride'],1]
                rate = [m['rate'],m['rate']]
                if (m['convType'] == "conv2d"):
                    x = self.conv(x,strinde,m['blockId'])
                    buff.append(x)
                elif (m['convType'] == "separableConv"):
                    x = self.separableConv(x,strinde,m['blockId'],rate)
                    buff.append(x)
        self.heatmaps = self.convToOutput(x, 'heatmap_2')
        self.offsets = self.convToOutput(x, 'offset_2')
        self.displacementFwd = self.convToOutput(x, 'displacement_fwd_2')
        self.displacementBwd = self.convToOutput(x, 'displacement_bwd_2')
        self.heatmaps = tf.sigmoid(self.heatmaps, 'heatmap')


    def start(self):
        isUser = False

        cap = cv2.VideoCapture(0) # Read camera
        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width_factor =  cap_width/self.width
        height_factor = cap_height/self.height
        
        config = tf.ConfigProto(device_count = {'GPU': 0})

        with tf.Session(config = config) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            save_dir = './checkpoints'
            save_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, save_path)
            flag, frame = cap.read()
            count = 0
            drawnsk = 0
            addedNotUser = False
            recognizeAgain = False
            startime = time.time()
            first_check = True

            coordinatesTime = None 
            startcoordinatesTime = time.time()

            leftHandpoint = []
            rectCenters = []
            objects = []

            leftHandT = []
            rightHandT = []

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter('output.avi', fourcc, 30, (int(cap_width), int(cap_height)))
            record = False
            
            
            while flag:

                orig_image = frame
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(float)
                frame = frame * (2.0 / 255.0) - 1.0
                frame = np.array(frame, dtype=np.float32)
                frame = frame.reshape(1, self.width, self.height, 3)
                heatmaps_result, offsets_result, displacementFwd_result, displacementBwd_result \
                    = sess.run([self.heatmaps, \
                                self.offsets, \
                                self.displacementFwd, \
                                self.displacementBwd], feed_dict={self.image: frame } )
                '''
                poses = decode_single_pose(heatmaps_result, offsets_result, 16, width_factor, height_factor)
                '''
                poses = decodeMultiplePoses(heatmaps_result, offsets_result, \
                                            displacementFwd_result, \
                                            displacementBwd_result, \
                                            width_factor, height_factor)
        
                x = self.get_only_poses_with_valid_score(poses)

                if len(x) > 0:
                    color = color_table[0]
                    #drawKeypoints(x[0], orig_image, color)
                    #rect = drawPositions(x[0], orig_image, color)

                    self.gestures.update(x[0])

                    # print(self.gestures.leftHandSwipeLeft)
                    # print(self.gestures.rightHandSwipeLeft)
                    # print(self.gestures.leftHandSwipeRight)
                    # print(self.gestures.rightHandSwipeRight)
                    # print(self.gestures.leftHandSwipeUp)
                    # print(self.gestures.rightHandSwipeUp)
                    # print(self.gestures.leftHandSwipeDown)
                    # print(self.gestures.rightHandSwipeDown)
                    # print("---------------------------------")

                    self.gestures.drawPoints(x[0], orig_image, color)
                        
                cv2.imshow("Hand Gesture Tracking", orig_image)

                flag, frame = cap.read() 
                count = count + 1

                mkey = cv2.waitKey(1)

                if mkey == ord('q'):
                    flag = False
                    break
                if mkey == ord('r'):
                    record = True

                if record:
                    out.write(orig_image)
        
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        
        
