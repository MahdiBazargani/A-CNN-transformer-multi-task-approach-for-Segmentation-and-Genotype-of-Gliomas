import argparse
from data.loader import Loader
import os
from data.landmarks import Landmarks
import math
import torchio as tio
from utils.tools import random_id
from utils.tools import logger
import logging
import pickle 
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data',metavar='',type=str,default='../MICCAI_BraTS2020_TrainingData',help='path to input images')
parser.add_argument('--output_path',metavar='',type=str,default='preprocessed4',help='path to save preprocessed images')
parser.add_argument('--landmarks_path',metavar='',type=str,default='landmarks',help='path to save landmarks')
args = parser.parse_args()

def transform(subject):

    _,x,y,z= subject.shape
    x = math.ceil((128-x)/2) 
    if x<0:
        x=0
    y = math.ceil((128-y)/2) 
    if y<0:
        y=0
    z = math.ceil((128-z)/2) 
    if z<0:
        z=0
    pad=tio.Pad((x,y,z))
    subject = pad(subject)
    return subject



def transform2(subject):

    label= tio.LabelMap(tensor=torch.where(torch.nn.functional.pad(subject,(10,10,10,10,10,10)) ==0,1,0))
    mask = tio.KeepLargestComponent()(label)['data']
    mask = mask[:,10:-10,10:-10,10:-10]
    foreground_mask = torch.where(mask==0,True,False)

    # label= tio.LabelMap(tensor=torch.where(torch.nn.functional.pad( subjects[i]['brain']['data'],(10,10,10,10,10,10) ) ==1,0,1))
    # print(tio.KeepLargestComponent()(label)['data'].sum().item()-label['data'].sum().item())

    # Assuming your image tensor is named 'image_tensor'
    # Shape: (100, 100, 100)
    # Convert the NumPy array to a PyTorch tensor (if it's not already)
    image_tensor = subject.float()
    # Extract foreground and background
    # foreground_mask = (image_tensor == 0)
    foreground = image_tensor[foreground_mask]
    background = image_tensor[~foreground_mask]


    # per=torch.quantile(foreground,torch.tensor([0.005,.995]))
    # foreground = torch.clip(foreground,*per)

    # Normalize foreground values
    normalized_foreground = (foreground - torch.mean(foreground)) / (torch.std(foreground))

    # Replace the normalized foreground values in the original tensor
    image_tensor[foreground_mask] = normalized_foreground
    


    # The background values will remain at zero

    # Now, 'image_tensor' contains the normalized values in the foreground while keeping the background values at zero.
    return image_tensor



def trainsforms(method,subjects):
    remapping={4:3}
    if method == 1:
        landmarks_dict = Landmarks(subjects,os.path.join(os.path.abspath(os.path.dirname(__file__)), args.landmarks_path))()
        return tio.Compose([
        tio.RemapLabels(remapping),
        tio.HistogramStandardization(landmarks_dict,
            masking_method=tio.ZNormalization.mean),        # standardize histogram of foreground
        tio.ZNormalization(
            masking_method=tio.ZNormalization.mean),        # zero mean, unit variance of foreground
        tio.CropOrPad(None,mask_name='brain'),
        tio.Lambda(transform),
    ])
    if method == 2:
        return tio.Compose([
        tio.RemapLabels(remapping),
        tio.ZNormalization(
            masking_method='brain'),        # zero mean, unit variance of foreground
        tio.CropOrPad(None,mask_name='brain'),
        tio.Lambda(transform),
    ])
    if method == 3:
        return tio.Compose([
        tio.RemapLabels(remapping),
        tio.Lambda(transform2,types_to_apply=[tio.INTENSITY]),
        tio.CropOrPad(None,mask_name='brain'),
        tio.Lambda(transform),
    ])


def main():

    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    data_loader=Loader(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.path_to_data))
    subjects=data_loader.get_subjects()

    transforms = trainsforms(3,subjects)

    output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_path )
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    logging.info(transforms)
    logging.info('total number of subjects: {}'.format(len(subjects)))
    for i in range(len(subjects)):
        logging.info('preprocessing subject {}'.format(i+1))
        subject=transforms(subjects[i])
        image_size = subjects[i]['t1'].shape
        padding=subject.get_inverse_transform()[0].padding
        path=os.path.join(output_path,os.path.dirname(subject['t1'].path).replace('/','\\').split('\\')[-1] )
        if not os.path.exists(path):
            os.mkdir(path)
        for image in subject.values():
            image.save(os.path.join(path,os.path.basename(image.path)))
            with open(os.path.join(path,'info.pkl'),'wb') as w:
                pickle.dump(padding,w)
                pickle.dump(image_size,w)
            
if __name__ == '__main__':
    run_id=random_id(20)    
    logger(os.path.join(os.path.abspath(os.path.dirname(__file__)),'logs','preprocessing_{}.txt'.format(run_id)))
    main()
    