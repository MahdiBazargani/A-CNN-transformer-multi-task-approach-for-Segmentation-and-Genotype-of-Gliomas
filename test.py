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


parser = argparse.ArgumentParser()

parser.add_argument('--epochs',metavar='', type=int, default=100, help='Number of epochs')
parser.add_argument('--random_state',metavar='', type=int, default=0, help='Batch size')
parser.add_argument('--path_to_data',metavar='',type=str,default='preprocessed4',help='path to images')
parser.add_argument('--run_id',metavar='',type=str,default='RTJMBCAFNMJYNOKIHGWV',help='path to images')
parser.add_argument('--cluster',metavar='', type=int, default=2, help='cluster')

args = parser.parse_args()

run_id=args.run_id

def inference(subject,model,device):
    patch_overlap = 64
    patch_size = 128,128,128
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=8)
    aggregator = tio.inference.GridAggregator(grid_sampler,'hann')
    # for model in models.values():
    model.eval()
    with torch.no_grad():
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
                outputs = model(inp)
                aggregator.add_batch(torch.flip(torch.sigmoid(outputs[-1]), dims=mode), locations)
                

    output_tensor = aggregator.get_output_tensor()
         
    return output_tensor




def main():

    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))

    data_loader=Loader(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.path_to_data))
    subjects=data_loader.get_subjects()

    train_test_split = TrainTestSplit(subjects,args.random_state)
    _, validation_subjects = train_test_split(args.cluster)

    validation_dataset=tio.SubjectsDataset(validation_subjects, transform=None)

    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

    logging.info('number of validation subjects: {}'.format(len(validation_dataset)))
    logging.info('run id: {}'.format(run_id))


    models=load_model(run_id,device)

    results=np.zeros(len(validation_dataset))
    results2=np.zeros(len(validation_dataset))
    results3=np.zeros(len(validation_dataset))

    for i in range(len(validation_dataset)):

        res=inference(validation_dataset[i],models,device)
        # print(res.shape)
        # print(res[0,:3,:3,:3])

        a=torch.where(res[0]>0.5,1,0)
        b=torch.where(res[1]>0.5,1,0)
        c=torch.where(res[2]>0.5,1,0)

        WT=a
        TC=torch.logical_and(a,b)
        ET =torch.logical_and(torch.logical_and(a,b),c)

        # res=torch.round(inference(test_dataset[i]))
        lbo=validation_dataset[i]['label']['data'][0,:,:,:]
        # res2=torch.zeros((4,240,240,155))
        # res2[1:,:,:,:]=res
        # res2=res2.argmax(dim=0)
        # a=torch.where(res2>0,1,0)
        b=torch.where(lbo>0,1,0)
        results[i]=Dice(WT,b).item()
        logging.info(results[i])

        # a=torch.where(res2==3,1,0)
        b=torch.where(lbo==3,1,0)
        # print(f'a={a.sum()},b={b.sum()}')
        results2[i]=Dice(ET,b).item()
        logging.info(results2[i])


        # a=torch.where(torch.logical_or(res2==3,res2==1) ,1,0)
        b=torch.where(torch.logical_or(lbo==3,lbo==1),1,0)
        # print(f'a={a.sum()},b={b.sum()}')
        results3[i]=Dice(TC,b).item()
        logging.info(results3[i])

    logging.info('')
    logging.info(results.mean())
    logging.info(results2.mean())
    logging.info(results3.mean())




if __name__ == '__main__':
    logger(os.path.join(os.path.abspath(os.path.dirname(__file__)),'logs','Testing_{}.txt'.format(run_id)))
    main()
    