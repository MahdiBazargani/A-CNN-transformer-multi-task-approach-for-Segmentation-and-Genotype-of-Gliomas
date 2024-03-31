import argparse
from data.loader import Loader
import os
import torchio as tio
import torch
from utils.tools import logger
import time
from utils.tools import random_id
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data',metavar='',type=str,default='../MICCAI_BraTS2020_ValidationData',help='path to input images')
parser.add_argument('--run_id',metavar='',type=str,default='',help='path to images')

args = parser.parse_args()

local_time = time.strftime('%m-%d-%Y %H:%M:%S', time.localtime())

def main():
    data_loader=Loader(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.path_to_data))
    subjects=data_loader.get_subjects(True)

    for i in range(len(subjects)):
        brain=tio.LabelMap(tensor=torch.where(subjects[i]['t1']['data']>0,1,0),affine=subjects[i]['t1'].affine)
        brain_mask_path = str(subjects[i]['t1'].path).replace('t1','brain')
        brain.save(brain_mask_path)
        logging.info('subject {} brain mask saved in {}'.format(i,brain_mask_path))

if __name__ == '__main__':
    run_id=random_id(20)    
    logger(os.path.join(os.path.abspath(os.path.dirname(__file__)),'logs','brain_mask_{}.txt'.format(run_id)))
    main()
    