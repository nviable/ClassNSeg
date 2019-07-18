"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing classification of ClassNSeg (the proposed method)
"""
#%%
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import face_recognition
from math import floor

#%%
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from model.ae import Encoder
from model.ae import ActivationLoss

#%%
video_path = 'predict.mp4'
imageSize = 256
gpu_id = 0
id = 100
out_path = 'predictions'
saved_weights_path = 'checkpoints/full'

#%%
def convert_locations(locations):
    t,r,b,l = locations
    dl_ = floor( (b-t + r-l) * 0.15 )
    e_ = floor((r-l-b+t)/2)
    return t - dl_, b + dl_, l - dl_ - e_, r + dl_ + e_

#%%
if __name__ == "__main__":

    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    # total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    locations = []
    faces = []
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    encoder = Encoder(3)
    encoder.load_state_dict(torch.load(os.path.join(saved_weights_path, 'encoder_' + str(id) + '.pt')))
    encoder.eval()  # get the best weights

    class Normalize_3D(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        
        def __call__(self, tensor):
            """
                Tensor: Normalized image.
            Args:
                tensor (Tensor): Tensor imag eof size (C, H, W) to be normalized)
            Returns:    """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor
    
    count = 0
    while success:
        face_batch = []
        face_locations = face_recognition.face_locations(frame)
        locations.append(face_locations)
        top, bottom, left, right = convert_locations(face_locations[0])
        face = frame[top:bottom,left:right, :]
        face = cv2.resize(face, (256,256))
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

        count += 1

    print(tol_pred)
    print(tol_pred_prob)
#%%
