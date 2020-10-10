import numpy as np
import torch
import os
import cv2
import math
import datetime

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    # folder == 'train' or 'val'
    def __init__(self, train_path, nfeatures, folder = 'train'):

        self.files = []
        self.folder = folder + '_data_rs'
        self.files += [train_path + f for f in os.listdir(os.path.join(train_path, self.folder))]

        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx], allow_pickle=True)
        file_name = self.files[idx]

        kp1 = data[0]
        kp2 = data[1]
        desc1 = data[2]
        desc2 = data[3]
        image1 = data[4]
        image2 = data[5]
        M = data[6]

        width, height = image1.shape[0], image1.shape[1]

        kp1_np = np.array([(kp[0], kp[1]) for kp in kp1]) # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
        kp2_np = np.array([(kp[0], kp[1]) for kp in kp2])
        scores1_np = np.array([kp[2] for kp in kp1])
        scores2_np = np.array([kp[2] for kp in kp2])

        if len(kp1) < 1 or len(kp2) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image1,
                'image1': image2,
                'file_name': file_name
            }
        matched = self.matcher.match(desc1, desc2)

        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] # why [0, :, :]

        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        visualize = False
        if visualize:
            matches_dmatch = []
            for idx in range(matches.shape[0]):
                dmatch = cv2.DMatch(matches[idx], min2[matches[idx]], 0.0)
                print("Match {matches[idx]} {min2[matches[idx]]} dist={dists[matches[idx], min2[matches[idx]]]}")
                matches_dmatch.append(dmatch)
            out = cv2.drawMatches(image, kp1, warped, kp2, matches_dmatch, None)
            cv2.imshow('a', out)
            cv2.waitKey(0)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        image = torch.from_numpy(image/255.).double()[None].cuda()
        warped = torch.from_numpy(warped/255.).double()[None].cuda()

        return{
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        }
