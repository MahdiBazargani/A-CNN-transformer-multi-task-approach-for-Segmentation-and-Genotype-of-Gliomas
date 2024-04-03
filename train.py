import argparse
import torch
import os
from data.loader import Loader
import torchio as tio

from models.experiment import Experiment

from tqdm import tqdm

from utils.tools import *


from utils.tools import logger
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--epochs',metavar='', type=int, default=250, help='Number of epochs')
parser.add_argument('--random_state',metavar='', type=int, default=0, help='Batch size')
parser.add_argument('--batch_size',metavar='', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate',metavar='', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--path_to_data',metavar='',type=str,default='preprocessed4',help='path to images')
parser.add_argument('--cluster',metavar='', type=int, default=4, help='cluster')

parser.add_argument('--resume',metavar='',type=bool,default=False,help='path to images')
parser.add_argument('--run_id',metavar='',type=str,default='',help='path to images')

parser.add_argument('--description',metavar='',type=str,default='nn uunet implemetation',help='path to images')


args = parser.parse_args()

if args.resume:
    run_id=args.run_id
else:
    run_id=random_id(20)

#training_transforms = tio.Compose([
    #tio.RandomElasticDeformation(p=0.5),
    # tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 10% of times
    #tio.OneOf({                                # either
    #    tio.RandomAffine(degrees=0): 0.5,               # random affine
    #    tio.RandomElasticDeformation(): 0.4,   # or random elastic deformation
    #    tio.RandomBiasField(coefficients=0.1):0.1,    # magnetic field inhomogeneity 30% of times
    #}, p=0.8),                                 # applied to 80% of images
    #tio.RandomGamma(p=0.2,log_gamma=(-0.05,0.05)),
    # tio.RandomFlip(axes=(0,1,2)),
#])

#training_transforms = tio.Compose([
#    tio.RandomAffine(scales=(0.65, 1.6), degrees=30 , p=0.3, isotropic =False,default_pad_value='mean'),
#    tio.RandomNoise(std=(0,0.1),p=0.15),
#    tio.RandomBlur(p=0.2,std=(0.5,1.5)),
#    tio.RandomElasticDeformation(p=0.3),
#    tio.RandomAnisotropy(p=0.1),
#    tio.RandomGamma(p=0.3),
#    tio.RandomFlip(axes=(0,1,2)),
#])
validation_transforms = tio.Compose([
    tio.RandomFlip(axes=(0,1,2)),
])


training_transforms = tio.Compose([
    tio.RandomAffine(scales=(0.8, 1.2), degrees=3 , p=0.3, isotropic =False,default_pad_value='mean'),
    tio.RandomNoise(std=(0,0.1),p=0.15),
    tio.RandomBlur(p=0.2,std=(0,1)),
    tio.RandomElasticDeformation(p=0.3),
    tio.RandomAnisotropy(p=0.1,downsampling=(1,2)),
    tio.RandomGamma(p=1,log_gamma=0.2),
    tio.RandomFlip(axes=(0,1,2)),
])


def main():
    logging.info('hello')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))

    data_loader=Loader(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.path_to_data))
    subjects=data_loader.get_subjects()

    train_test_split = TrainTestSplit(subjects,args.random_state)
    train_subjects, validation_subjects = train_test_split(args.cluster)

    train_dataset=tio.SubjectsDataset(train_subjects, transform=training_transforms)
    validation_dataset=tio.SubjectsDataset(validation_subjects, transform=validation_transforms)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    logging.info('number of train subjects: {}'.format(len(train_dataset)))
    logging.info('number of validation subjects: {}'.format(len(validation_dataset)))
    logging.info('run id: {}'.format(run_id))
    experiment = Experiment(train_dataset,validation_dataset,run_id,device,args.learning_rate,args.epochs)
    logging.info('Experiment started')
    experiment.start()


if __name__ == '__main__':

    logger(os.path.join(os.path.abspath(os.path.dirname(__file__)),'logs','Experiment_{}.txt'.format(run_id)))
    main()
    