# camera-ready

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(ROOT)
from datasets import DatasetFullSeq # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append(os.path.join(ROOT, "model"))
from deeplabv3 import DeepLabV3

sys.path.append(os.path.join(ROOT, "utils"))
from utils import label_img_to_color

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import tqdm

batch_size = 256
overlay_iterations = 19

network = DeepLabV3("eval_full_seq", project_dir=ROOT).cuda()
network.load_state_dict(torch.load(os.path.join(ROOT, "pretrained_models/model_13_2_2_2_epoch_580.pth")))

os.makedirs(os.path.abspath(os.path.join(network.model_dir, "..", "model_eval_seq_seg")), exist_ok=True)

sequence_path = os.path.join(ROOT, "data/sequence")

test_dataset = DatasetFullSeq(sequence_data_path=sequence_path, mode="test")
val_dataset = DatasetFullSeq(sequence_data_path=sequence_path, mode="val")
train_dataset = DatasetFullSeq(sequence_data_path=sequence_path)

datasets = {"test":test_dataset, "val":val_dataset, "train":train_dataset}
data_loaders = {}

for key, dataset in datasets.items():
    print ("num {} batches:".format(key), int(len(dataset)/batch_size))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=1)
    data_loaders[key] = data_loader

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
# unsorted_img_ids = []
overlay_count = 0
for mode, data_loader in data_loaders.items():
    for step, (imgs, img_ids) in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))

            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            ########################################################################
            # save data for visualization:
            ########################################################################
            outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
            pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            for i in range(pred_label_imgs.shape[0]):
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_id = img_ids[i]
                cv2.imwrite(os.path.abspath(os.path.join(network.model_dir, "..", "model_eval_seq_seg", img_id + "_seg.png")), pred_label_img)

                if overlay_count % overlay_iterations == 0:
                    img = imgs[i] # (shape: (3, img_h, img_w))

                    img = img.data.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                    img = img*np.array([0.229, 0.224, 0.225])
                    img = img + np.array([0.485, 0.456, 0.406])
                    img = img*255.0
                    img = img.astype(np.uint8)

                    pred_label_img_color = label_img_to_color(pred_label_img)
                    overlayed_img = 0.35*img + 0.65*pred_label_img_color
                    overlayed_img = overlayed_img.astype(np.uint8)

                    img_h = overlayed_img.shape[0]
                    img_w = overlayed_img.shape[1]

                    cv2.imwrite(os.path.join(network.model_dir, img_id + "_overlayed.png"), overlayed_img)
                    # cv2.imwrite(network.model_dir + "/" + img_id + ".png", img)
                    # cv2.imwrite(network.model_dir + "/" + img_id + "_pred.png", pred_label_img_color)
                    # cv2.imwrite(network.model_dir + "/" + img_id + "_overlayed.png", overlayed_img)
                overlay_count += 1
                # unsorted_img_ids.append(img_id)

################################################################################
# create visualization video:
################################################################################
# out = cv2.VideoWriter("%s/thn_combined.avi" % network.model_dir, cv2.VideoWriter_fourcc(*"MJPG"), 12, (2*img_w, 2*img_h))
# sorted_img_ids = sorted(unsorted_img_ids)
# for img_id in sorted_img_ids:
#     img = cv2.imread(network.model_dir + "/" + img_id + ".png", -1)
#     pred_img = cv2.imread(network.model_dir + "/" + img_id + "_pred.png", -1)
#     overlayed_img = cv2.imread(network.model_dir + "/" + img_id + "_overlayed.png", -1)

#     combined_img = np.zeros((2*img_h, 2*img_w, 3), dtype=np.uint8)

#     combined_img[0:img_h, 0:img_w] = img
#     combined_img[0:img_h, img_w:(2*img_w)] = pred_img
#     combined_img[img_h:(2*img_h), (int(img_w/2)):(img_w + int(img_w/2))] = overlayed_img

#     out.write(combined_img)

# out.release()
