import os
import torch
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse
import imutils
import cv2
from PIL import Image
from model.ae import Encoder
from model.ae import Decoder
from model.ae import ActivationLoss
from model.ae import SegmentationLoss
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--test_video', default ='face_extraction/demo/Input/Obama.avi', help='path to input video')
# parser.add_argument('--fps', type=float, default = 30, help='FPS')
parser.add_argument('--out_video', default ='face_extraction/demo/Output/Obama.mp4', help='path to output video')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--id', type=int, default=46, help="checkpoint ID")
parser.add_argument('--outf', default='FF-256-Full-Seg_v7/easy_c', help='folder to output images and model checkpoints')
parser.add_argument('--prototxt', default='face_extraction/deploy.prototxt', help='path to Caffe deploy prototxt file')
parser.add_argument('--model', default='face_extraction/res10_300x300_ssd_iter_140000.caffemodel', help='path to Caffe pre-trained model')
parser.add_argument('--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
parser.add_argument('--overlay', type=int, default=0, help='whether to output overlayed or sidebyside video')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    face_detection_net = cv2.dnn.readNetFromCaffe(opt.prototxt, opt.model)
    t = time()
    encoder = Encoder(3)
    decoder = Decoder(3)

    encoder.load_state_dict(torch.load(os.path.join(opt.outf,'encoder_' + str(opt.id) + '.pt')))
    encoder.eval()
    decoder.load_state_dict(torch.load(os.path.join(opt.outf,'decoder_' + str(opt.id) + '.pt')))
    decoder.eval()

    if opt.gpu_id >= 0:
        encoder.cuda(opt.gpu_id)
        decoder.cuda(opt.gpu_id)

    transform_fwd = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    ##################################################################################

    vc = cv2.VideoCapture(opt.test_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vr = None

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
     
    if int(major_ver)  < 3 :
        fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = vc.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    preds = []
    count = 1
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=1000, height=1000)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        if (vr == None):
            _w = w*2 if (opt.overlay == 0) else w
            vr = cv2.VideoWriter(opt.out_video, fourcc, fps, (_w,h))

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

        # pass the blob through the network and obtain the detections and predictions
        face_detection_net.setInput(blob)
        detections = face_detection_net.forward()

        crop_img_lst = []
        boxes = []

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < opt.confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")
            boxes.append(box)

        if len(boxes) == 0:
            continue
        
        print("Frame: %i" % count)
        count += 1

        # elif len(boxes) > 1:
        #     boxes = non_max_suppression_slow(np.stack(boxes, axis=0))

        # loop over the detections
        for i in range(0, len(boxes)):
            (startX, startY, endX, endY) = boxes[i]

            # convert rectangle box to square
            width = endX - startX
            height = endY - startY

            if height > width:
                delta = int((height - width)/2)
                startX = startX - delta
                endX = endX + delta

            elif width < height:
                delta = int((width - height)/2)
                startY = startY - delta
                endY = endY + delta

            if startX < 0:
                startX = 0
            if endX >= w:
                endX = w
            if startY < 0:
                startY = 0
            if endY >= h:
                endXY = h

            # crop face areas

            crop_img = frame[startY:endY, startX:endX]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            crop_img_lst.append(((crop_img), (startX, startY, endX, endY)))

        if len(crop_img_lst) > 0:
            img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)
            for i in range(len(crop_img_lst)):
                img_data = transform_fwd(Image.fromarray(crop_img_lst[i][0])).unsqueeze(0)
                img_tmp = torch.cat((img_tmp, img_data), dim=0)

            if opt.gpu_id >= 0:
                img_tmp = img_tmp.cuda(opt.gpu_id)

            latent = encoder(img_tmp).reshape(-1, 2, 64, 16, 16)

            zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)

            output_pred = torch.zeros(img_tmp.shape[0])

            if one[0] >= zero[0]:
                output_pred[0] = 1.0
                preds.append(one[0].item())
            else:
                output_pred[0] = 0.0
                preds.append(1-zero[0].item())

            y = torch.eye(2)

            if opt.gpu_id >= 0:
                output_pred = output_pred.cuda(opt.gpu_id)
                y = y.cuda(opt.gpu_id)

            y = y.index_select(dim=0, index=output_pred.data.long())

            latent = (latent * y[:,:,None, None, None]).reshape(-1, 128, 16, 16)

            seg, rect = decoder(latent)

            seg = seg[:,1,:,:].detach().cpu()
            seg[seg >= 0.5] = 255.0
            seg[seg < 0.5] = 0.0

            seg = seg.detach().cpu().numpy()

            seg_idx = -1
            seg_max = 0

            for i in range(output_pred.shape[0]):
                (startX, startY, endX, endY) = crop_img_lst[i][1]

                if( opt.overlay == 0 ):
                    if output_pred[i] >= 0.5:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)
                        cv2.putText(frame, 'Real', (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        #cv2.putText(frame, 'Real', (startX + int(delta * 0.25), endY + int(delta * 1.35)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
                        cv2.putText(frame, 'Fake', (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        #cv2.putText(frame, 'Fake', (startX + int(delta * 0.25), endY + int(delta * 1.35)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                seg_sum = seg[i].sum()
                if seg_sum > seg_max:
                    seg_max = seg_sum
                    seg_idx = i


        blank_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        if seg_idx >= 0:
            (startX, startY, endX, endY) = crop_img_lst[seg_idx][1]
            mask = seg[seg_idx].astype(np.uint8)
            mask = cv2.resize(mask, (endX - startX, endY - startY))
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
            fill_mask = mask if (opt.overlay == 0) else np.zeros(np.shape(mask))
            mask = np.concatenate((fill_mask, fill_mask, mask), axis=2)
            blank_img[startY:endY, startX:endX, :] = blank_img[startY:endY, startX:endX, :] + mask
        
        frame = np.concatenate((frame, blank_img), axis=1) if (opt.overlay == 0 ) else cv2.addWeighted(frame, 1, blank_img, 0.5, 0)

        # cv2.imshow("Face detector from camera stream", frame)
        # key = cv2.waitKey(1) & 0xFF
        #
        # if key == ord("q"):
        #     break

        vr.write(frame)
        #vr.write(blank_img)

    total_t = time() - t
    print("Total frames {} | Total time {} | Average {} | FPS: {}".format(count, total_t, total_t/count, fps))

    vc.release()
    vr.release()
