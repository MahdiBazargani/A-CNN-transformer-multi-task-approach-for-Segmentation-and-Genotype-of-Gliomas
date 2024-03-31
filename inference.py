import argparse
import torch
import os
from data.loader import Loader
import torchio as tio
from utils.tools import *

import numpy as np
from utils.tools import logger
import logging

from models.experiment import load_model
from models.criterions import Dice

import math
parser = argparse.ArgumentParser()

parser.add_argument('--epochs',metavar='', type=int, default=100, help='Number of epochs')
parser.add_argument('--random_state',metavar='', type=int, default=0, help='Batch size')
parser.add_argument('--path_to_data',metavar='',type=str,default='../MICCAI_BraTS2020_ValidationData',help='path to images')
parser.add_argument('--run_ids',metavar='',type=str,default='DTUBPINILYETBKERNWJS,TVPGKEESSRHQSQXVPXYF,CLOAGMAGZSYVZZRAXERO,RIFXYGYYUYNRCQHGAJGE,RTJMBCAFNMJYNOKIHGWV',help='path to images')


args = parser.parse_args()

def get_run_ids(run_ids):
    res=[]
    for item in run_ids.split(','):
        res.append(item.strip())
    return res

run_ids= get_run_ids(args.run_ids)



def inference(subject,all_models,device):
    patch_overlap = 64
    patch_size = 128,128,128
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=8)
    aggregator = tio.inference.GridAggregator(grid_sampler,'hann')

    with torch.no_grad():
        for models in all_models:

            for patches_batch in patch_loader:
                t1 = patches_batch['t1'][tio.DATA]  # key 't1' is in subject
                t2 = patches_batch['t2'][tio.DATA]  # key 't2' is in subject
                t1ce = patches_batch['t1ce'][tio.DATA]  # key 't1ce' is in subject
                flair = patches_batch['flair'][tio.DATA]  # key 'flair' is in subject
                inputs=torch.cat([t1,t2,t1ce,flair],dim=1)
                locations = patches_batch[tio.LOCATION]
                inputs=inputs.to(device=device)
                modes = [(),(2,),(3,),(4,),(2,3),(2,4),(3,4),(2,3,4)]
                for mode in modes:
                    inp = torch.flip(inputs, dims=mode)    
                    outputs = models(inp)
                    aggregator.add_batch(torch.flip(torch.sigmoid(outputs[-1]), dims=mode), locations)
                    

    output_tensor = aggregator.get_output_tensor()
         
    return output_tensor



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
    return image_tensor



remapping={4:3}
transforms= tio.Compose([
    tio.RemapLabels(remapping),
    tio.Lambda(transform2,types_to_apply=[tio.INTENSITY]),
    tio.CropOrPad(None,mask_name='brain'),
    tio.Lambda(transform),
])


def main():

    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))

    data_loader=Loader(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.path_to_data))
    subjects=data_loader.get_subjects(test_data=True)


    dataset=tio.SubjectsDataset(subjects, transform=transforms)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('number of validation subjects: {}'.format(len(dataset)))
    logging.info('run id: {}'.format(run_ids))

    all_models=[]
    for run_id in run_ids:
        all_models.append(load_model(run_id,device))

    for models in all_models:
        # for model in models.values():
        models.eval()

    # for subject in validation_dataset:
    #     inference(subject,models,device)


    # results=np.zeros(len(validation_dataset))
    # results2=np.zeros(len(validation_dataset))
    # results3=np.zeros(len(validation_dataset))


    for i in range(0,len(dataset)):
        data=dataset[i]
        a=inference(data,all_models,device)
        c=data.get_inverse_transform()[0]
        
        WT=torch.where(a[0]>0.5,1,0)
        TC=torch.where(a[1]>0.5,1,0)
        ET=torch.where(a[2]>0.5,1,0)
        final=torch.zeros_like(WT)#.unsqueeze(0)
        final[torch.where(torch.logical_and(torch.logical_and(WT,TC),ET))]=4
        final[torch.where(torch.logical_and(torch.logical_and(WT,TC),1-ET))]=1
        final[torch.where(torch.logical_and(torch.logical_and(WT,1-TC),1-ET))]=2
        final=final.unsqueeze(0)
        # print(c.padding)
        ww=final.shape[1]+c.padding[0]+c.padding[1]-240
        if ww>0:
            ww=int(ww/2)
        ww2=final.shape[2]+c.padding[2]+c.padding[3]-240
        if ww2>0:
            ww2=int(ww2/2)
        ww3=final.shape[3]+c.padding[4]+c.padding[5]-155
        if ww3>0:
            ww3=int(ww3/2)

        crop= tio.Crop((ww,ww2,ww3))
        final=crop(final)
        fres=c(final)
        logging.info(fres.shape)
        if not os.path.exists('results2'):
            os.mkdir('results2')
        #path='results/BraTS20_Validation_{:03d}.nii.gz'.format(i+1)
        path=  'results2/BraTS20_Validation_{}.nii.gz'.format(str(subjects[i]['t1'].path).split('/')[-1].split('_')[-2])   
        tio.LabelMap(tensor=fres,affine=subjects[i]['t1'].affine).save(path)


    # for i in range(len(dataset)):

    #     res = inference(dataset[i],all_models,device)
    # #     res= re/re2

    #     a=torch.where(res[0]>0.5,1,0)
    #     b=torch.where(res[1]>0.5,1,0)
    #     c=torch.where(res[2]>0.5,1,0)

    #     WT=a
    #     TC=torch.logical_and(a,b)
    #     ET =torch.logical_and(torch.logical_and(a,b),c)

    #     # res=torch.round(inference(test_dataset[i]))
    #     lbo=validation_dataset[i]['label']['data'][0,:,:,:]
    #     # res2=torch.zeros((4,240,240,155))
    #     # res2[1:,:,:,:]=res
    #     # res2=res2.argmax(dim=0)
    #     # a=torch.where(res2>0,1,0)
    #     b=torch.where(lbo>0,1,0)
    #     results[i]=Dice(WT,b).item()
    #     logging.info(results[i])

    #     # a=torch.where(res2==3,1,0)
    #     b=torch.where(lbo==3,1,0)
    #     # print(f'a={a.sum()},b={b.sum()}')
    #     results2[i]=Dice(ET,b).item()
    #     logging.info(results2[i])


    #     # a=torch.where(torch.logical_or(res2==3,res2==1) ,1,0)
    #     b=torch.where(torch.logical_or(lbo==3,lbo==1),1,0)
    #     # print(f'a={a.sum()},b={b.sum()}')
    #     results3[i]=Dice(TC,b).item()
    #     logging.info(results3[i])

    #     logging.info('')
    # logging.info(results.mean())
    # logging.info(results2.mean())
    # logging.info(results3.mean())




if __name__ == '__main__':
    logger(os.path.join(os.path.abspath(os.path.dirname(__file__)),'logs','Inference_{}.txt'.format(1)))
    main()
    