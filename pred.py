"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing classification of ClassNSeg (the proposed method)
"""
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import face_recognition
from math import floor
import argparse

import torch.utils.data
from model.ae import Encoder
from model.ae import ActivationLoss

parser = argparse.ArgumentParser()
parser.add_argument('--input', default ='predict.mp4', help='input file path')
opt = parser.parse_args()

class Predictor(object):
    def __init__(self, video_path, imageSize=256, gpu_id=0 ):
        self.path = video_path

    """
        Zoom out of the image and Convert locations from Top, Right, Bottom, Left to Top, Bottom, Left, Right
    """    
    def convert_locations(self, locations):
        t,r,b,l = locations
        dl_ = floor( (b-t + r-l) * 0.15 )
        e_ = floor((r-l-b+t)/2)
        return t - dl_, b + dl_, l - dl_ - e_, r + dl_ + e_

    """
        Predict using ClassNSeg method
    """
    def predictClassNseg(self, weight_path=os.path.join('checkpoints/full', 'encoder_' + str(100) + '.pt'), imageSize=256, gpu_id=0):
        vidcap = cv2.VideoCapture(self.path)
        success, frame = vidcap.read()

        # total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        locations = []
        faces = []
        tol_pred = np.array([], dtype=np.float)
        tol_pred_prob = np.array([], dtype=np.float)

        encoder = Encoder(3)
        if gpu_id >= 0:
            encoder.cuda(gpu_id)

        encoder.load_state_dict(torch.load(weight_path))
        encoder.eval()  # get the best weights

        count = 1
        print("\u25b8 Preprocessing")
        while success:  
            face_locations = face_recognition.face_locations(frame)
            locations.append(face_locations)
            top, bottom, left, right = self.convert_locations(face_locations[0])
            face = frame[top:bottom,left:right, :]
            face = cv2.resize(face, (256,256))
            faces.append(face)
            count +=1
            success, frame = vidcap.read()        
        print("\u2713 Finished Preprocessing {} frames".format(count))

        print("\u25b8 Beginning Training")
        for face in tqdm(faces):
            face_tensor = torch.from_numpy(np.array(face, np.float32, copy=False))
            face_tensor = face_tensor.transpose(0,1).transpose(0,2).contiguous()
            face_tensor = torch.unsqueeze(face_tensor, 0)

            if gpu_id >= 0:
                face_tensor = face_tensor.cuda(gpu_id)

            latent = encoder(face_tensor).reshape(-1, 2, 64, 16, 16)
            
            zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)

            output_pred = np.zeros((face_tensor.shape[0]), dtype=np.float)

            for i in range(face_tensor.shape[0]):
                if one[i] >= zero[i]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_pred = np.concatenate((tol_pred, output_pred))

            pred_prob = torch.softmax(torch.cat((zero.reshape(zero.shape[0],1), one.reshape(one.shape[0],1)), dim=1), dim=1)
            tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.cpu().numpy()))

        print(tol_pred)
        print(tol_pred_prob)

pred = Predictor(opt.input)
pred.predictClassNseg()