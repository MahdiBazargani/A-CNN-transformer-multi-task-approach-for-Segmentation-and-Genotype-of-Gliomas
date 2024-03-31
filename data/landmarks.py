import os
from pathlib import Path
import torchio as tio
import numpy as np

class Landmarks:

    def __init__(self,subjects,landmarks_path):
        self.subjects=subjects
        self.landmarks_path=landmarks_path
    def get_paths(self,modality):
        paths=[]
        for subject in self.subjects:
            paths.append(subject[modality].path)
        return paths

    def get_landmark(self,modality):
        paths = self.get_paths(modality)
        if not os.path.exists(self.landmarks_path):
            os.mkdir(self.landmarks_path)
        landmarks_path = Path(os.path.join(self.landmarks_path,'{}_landmarks.npy'.format(modality)))
        if landmarks_path.is_file():
            landmarks=np.load(landmarks_path)
        else:
            landmarks=tio.HistogramStandardization.train(paths,masking_function=tio.ZNormalization.mean)
            np.save(landmarks_path, landmarks)

        return landmarks
    def __call__(self):
        landmarks_dict = {
            't1': self.get_landmark('t1'),
            't2': self.get_landmark('t2'),
            't1ce': self.get_landmark('t1ce'),
            'flair': self.get_landmark('flair'),
        }
        return landmarks_dict