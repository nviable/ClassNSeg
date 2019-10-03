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
import torchvision.transforms as transforms
from PIL import Image

import torch.utils.data
from model.ae import Encoder
from model.ae import ActivationLoss

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default ='predict.mp4', help='input file path')
parser.add_argument('--step', '-s', default ='100', help='weights step to use')
opt = parser.parse_args()

class Normalize_3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
            Tensor: Normalized image.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Predictor(object):
    def __init__(self, video_path, imageSize=256, gpu_id=0 ):
        self.path = video_path
        self.transform_fwd = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

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
    def predictClassNseg(self, weight_path=os.path.join('checkpoints/full', 'encoder_' + str(opt.step) + '.pt'), imageSize=256, gpu_id=0):
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
        print("\u25b8 Preprocessing {}".format(self.path))
        while success:  
            face_locations = face_recognition.face_locations(frame)
            locations.append(face_locations)
            top, bottom, left, right = self.convert_locations(face_locations[0])
            face = frame[top:bottom,left:right, :]
            faces.append(face)
            count +=1
            success, frame = vidcap.read()        
        print("\u2713 Finished Preprocessing {} frames".format(count))

        print("\u25b8 Predicting Frames")
        img_tmp = torch.FloatTensor([]).view(0, 3, 256, 256)
        for face in tqdm(faces):
            face_tensor = self.transform_fwd(Image.fromarray(face)).unsqueeze(0)
            img_tmp = torch.cat((img_tmp, face_tensor), dim=0)

        if gpu_id >= 0:
            img_tmp = img_tmp.cuda(gpu_id)

        pre_latent = encoder(img_tmp)            
        latent = pre_latent.reshape(-1, 2, 64, 16, 16)

        zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)

        output_pred = torch.zeros(img_tmp.shape[0])

        for i in range(output_pred.shape[0]):
            if one[i] >= zero[i]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0
            
        print(output_pred)

pred = Predictor(opt.input)
pred.predictClassNseg()