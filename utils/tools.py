
from typing import Any
from sklearn.model_selection import KFold
import logging
import os
import random, string
import io


def random_id(length):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(length))


class TrainTestSplit:
    def __init__(self,subjects,random_state):
        kf = KFold(n_splits=5,shuffle=True,random_state=random_state)
        self.indices=list(kf.split(subjects))
        self.subjects = subjects
    def __call__(self, fold_number):
        
        train_subjects=[]
        for index in self.indices[fold_number][0]:
            train_subjects.append(self.subjects[index])
        validation_subjects=[]
        for index in self.indices[fold_number][1]:
            validation_subjects.append(self.subjects[index])    
        return train_subjects, validation_subjects
    
def logger(log_path):

    
    print(log_path)
    if not os.path.exists(os.path.dirname(log_path)):
        os.mkdir(os.path.dirname(log_path))
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(levelname)s][%(asctime)s] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')

    # args FileHandler to save log file
    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)

    # args StreamHandler to print log to console
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

