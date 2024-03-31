from models.unet.unet_model import UNet
import torch
import json
import os
import torch.optim as optim
from models.criterions import Loss
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import torch.nn as nn
#from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch.cuda.amp.grad_scaler import GradScaler

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.reshape(combined_shape)


def load_model(run_id,device):
    models=UNet(4,3)
    

    #models={'segmentation':PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 3,
    #                    (2, 2, 2, 2, 2), False, nn.InstanceNorm3d, None, None, None, nn.ReLU, deep_supervision=False)}



        # model= nn.DataParallel(model)
    models.to(device=device)
    models = torch.compile(models)


    checkpoint_path='checkpoints/'
    run_path=os.path.join(checkpoint_path,run_id)
    if os.path.exists(run_path):
        info_path=os.path.join(run_path,'info.json')
        if os.path.exists(info_path):
            f = open(info_path)
            info = json.load(f) 
            starting_epoch=info['starting_epoch']
            # for model in models.keys():
            model_path = os.path.join(run_path,f'{starting_epoch}.pth')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                models.load_state_dict(state_dict['model'])
    return models


class Experiment:


    # def _get_deep_supervision_scales(self):
    #     deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
    #         self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]

    #     return deep_supervision_scales

    def __init__(self,train_dataset,validation_dataset,run_id,device,learning_rate,num_epochs) -> None:
        
        self.counter=0
        loss = DC_and_BCE_loss({},
                                {'batch_dice': False,
                                'do_bg': True, 'smooth': 1e-5, 'ddp': False},
                                use_ignore_label=False,
                                dice_class=MemoryEfficientSoftDiceLoss)


        if True:
            # deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(5)])
            # if self.is_ddp and not self._do_i_compile():
            # if True:
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                # weights[-1] = 1e-6
            # else:
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            self.loss = DeepSupervisionWrapper(loss, weights)




        self.scaler = GradScaler()
        self.device=device
        self.train_dataset=train_dataset
        self.validation_dataset=validation_dataset
        self.num_epochs=num_epochs
        self.train_patch_loader=self._patch_loader(self.train_dataset,128,100,2,20,2)
        self.val_patch_loader=self._patch_loader(self.validation_dataset,128,100,1,20,1)

        self.model= UNet(4,3)

        # self.models={'segmentation':PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 3,
        #                     (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False)}

        # for model in self.models.values():
            #model= nn.DataParallel(model)
        self.model.to(device=self.device)
        self.model = torch.compile(self.model)


        checkpoint_path='checkpoints/'
        self.criterion = Loss() 
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)        
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.01,momentum=0.99,nesterov = True,weight_decay=3e-5)
        self.lr_scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer,total_iters=250, power=0.9)

        

        self.run_path=os.path.join(checkpoint_path,run_id)
        if not os.path.exists(self.run_path):
            # os.mkdir(self.run_path)
            self.starting_epoch=0
            self.min_val_loss=10000

        else:
            info_path=os.path.join(self.run_path,'info.json')
            if os.path.exists(info_path):
                f = open(info_path)
                info = json.load(f) 
                self.starting_epoch=info['starting_epoch']
                self.min_val_loss=info['min_val_loss']

                for model in self.models.keys():
                        model_path = os.path.join(self.run_path,f'{model}_{self.starting_epoch}.pth')
                        if os.path.exists(model_path):
                            state_dict = torch.load(model_path, map_location=self.device)
                            self.models[model].load_state_dict(state_dict['model'])
                            #self.optimizer[model].load_state_dict(state_dict['optimizer'])
                            #self.lr_scheduler[model].load_state_dict(state_dict['lr_scheduler'])

                            # print(1)

        #for optimizer_ in self.optimizer.values():
        #    optimizer_.param_groups[0]['lr']=learning_rate


    def _train(self):
        train_loss = 0
        train_dice = 0
        self.model.train()

        for batch_idx, patches_batch in enumerate(tqdm(self.train_patch_loader)):
            t1 = patches_batch['t1'][tio.DATA]  # key 't2' is in subject
            t2 = patches_batch['t2'][tio.DATA]  # key 't2' is in subject
            t1ce = patches_batch['t1ce'][tio.DATA]  # key 't1ce' is in subject
            flair = patches_batch['flair'][tio.DATA]  # key 'flair' is in subject
            targets = patches_batch['label'][tio.DATA]  # key 'brain' is in subject
            inputs=torch.cat([t1,t2,t1ce,flair],dim=1)

            
            uni = torch.unique(targets)
            if len(uni)==1:
                print(uni)
                print(self.counter)
                self.counter+=1
            
            # print(inputs.shape)
            # print(inputs.mean())
            # print(inputs.std())
            # print()
            inputs=inputs.to(self.device,non_blocking=True)
            # inputs.requires_grad_()
            targets=targets.to(self.device,non_blocking=True)

            # for optimizer_ in self.optimizer.values():
            self.optimizer.zero_grad(set_to_none=True)



            target= []
            self.pool= nn.AvgPool3d(kernel_size=1)
            WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
            TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
            ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
            target.append(torch.cat([WT,TC,ET],dim=1).float())

            self.pool= nn.AvgPool3d(kernel_size=2)
            WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
            TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
            ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
            target.append(torch.cat([WT,TC,ET],dim=1).float())

            self.pool= nn.AvgPool3d(kernel_size=4)
            WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
            TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
            ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
            target.append(torch.cat([WT,TC,ET],dim=1).float())

            self.pool= nn.AvgPool3d(kernel_size=8)
            WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
            TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
            ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
            target.append(torch.cat([WT,TC,ET],dim=1).float())

            self.pool= nn.AvgPool3d(kernel_size=16)
            WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
            TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
            ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
            target.append(torch.cat([WT,TC,ET],dim=1).float())

            
            # for item in target:
                # print(item.shape)
                # print(torch.unique(item,return_counts=True))

            with torch.autocast(device_type='cuda'):
                output=self.model(inputs)
                # for item in output:
                #     print(item.shape)
                # print(torch.unique(item,return_counts=True))

                # output = torch.sigmoid(output)
                loss = self.loss(output[::-1],target)
                #loss = self.criterion(output[0], targets , 8)
                #loss += 2*self.criterion(output[1], targets, 4)
                #loss += 4*self.criterion(output[2], targets, 2)
                # loss = 1*self.criterion(output[3], targets, 1)
                #loss /= 1

                
                train_dice+=self.criterion.Dice(output[-1].detach(), targets).item()

            self.scaler.scale(loss).backward()
            # for optimizer_ in self.optimizer.values():
            self.scaler.unscale_(self.optimizer)

            # for model in self.models.values():
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),12)

            ll = loss.detach().cpu().item()
            # print(ll)
            train_loss += ll
            # loss.backward()
            # for optimizer_ in self.optimizer.values():
                # optimizer_.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        

        # for lr_scheduler_ in self.lr_scheduler.values():
        self.lr_scheduler.step()

        train_loss /= len(self.train_patch_loader)
        train_dice /= len(self.train_patch_loader)

        return train_loss,train_dice



    def _pool(self,data):
        # return self.pool(data.float())
        return torch.where(self.pool(data.float())>0.5,1,0)
    
    def _validation(self,):
        val_loss = 0
        val_dice = 0

        # for model in self.models.values():
        self.model.eval()

        with torch.no_grad():
            for batch_idx, patches_batch in enumerate(tqdm(self.val_patch_loader)):
                t1 = patches_batch['t1'][tio.DATA]  # key 't2' is in subject
                t2 = patches_batch['t2'][tio.DATA]  # key 't2' is in subject
                t1ce = patches_batch['t1ce'][tio.DATA]  # key 't1ce' is in subject
                flair = patches_batch['flair'][tio.DATA]  # key 'flair' is in subject
                targets = patches_batch['label'][tio.DATA]  # key 'brain' is in subject
                inputs=torch.cat([t1,t2,t1ce,flair],dim=1)


                inputs=inputs.to(self.device,non_blocking=True)#.requires_grad()
                # inputs.requires_grad_()

                targets=targets.to(self.device,non_blocking=True)


                target= []
                self.pool= nn.AvgPool3d(kernel_size=1)
                WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
                TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
                ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
                target.append(torch.cat([WT,TC,ET],dim=1).float())

                self.pool= nn.AvgPool3d(kernel_size=2)
                WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
                TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
                ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
                target.append(torch.cat([WT,TC,ET],dim=1).float())

                self.pool= nn.AvgPool3d(kernel_size=4)
                WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
                TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
                ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
                target.append(torch.cat([WT,TC,ET],dim=1).float())

                self.pool= nn.AvgPool3d(kernel_size=8)
                WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
                TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
                ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
                target.append(torch.cat([WT,TC,ET],dim=1).float())

                self.pool= nn.AvgPool3d(kernel_size=16)
                WT = self._pool(torch.where(targets[:,0]>0,1,0)).unsqueeze(1)
                TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0)).unsqueeze(1)
                ET = self._pool(torch.where(targets[:,0]==3 ,1,0)).unsqueeze(1)
                target.append(torch.cat([WT,TC,ET],dim=1).float())
                

                # with torch.autocast(device_type='cuda'):
                with torch.autocast(device_type='cuda'):

                    output=self.model(inputs)
                    # output = torch.sigmoid(output)                
                    #loss = self.criterion(output[0], targets , 8)
                    #loss += 2*self.criterion(output[1], targets, 4)
                    #loss += 4*self.criterion(output[2], targets, 2)
                    # loss = 1*self.criterion(output[3], targets, 1)
                    #loss /= 15

                    loss = self.loss(output[::-1],target)


                    val_dice+=self.criterion.Dice(output[-1].detach(), targets).item()

                val_loss += loss.detach().cpu().item()


        val_loss /= len(self.val_patch_loader)
        val_dice /= len(self.val_patch_loader)

        return val_loss,val_dice



    def start(self):
        self.starting_epoch=0
        for epoch_index in range(self.starting_epoch,self.num_epochs):
            logging.info('Epoch {}/{}'.format(epoch_index+1, self.num_epochs))
            train_loss,train_dice = self._train()
            logging.info('Train loss: {:.6f}, Train dice score: {:.6f}'.format(train_loss, train_dice))

            val_loss,val_dice = self._validation()
            logging.info('Validation loss: {:.6f}, Validation dice score: {:.6f}'.format(val_loss, val_dice))

            if not os.path.exists(self.run_path):
                os.mkdir(self.run_path)

            # for model in self.models.keys():
            torch.save({
                'lr_scheduler':self.lr_scheduler.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'model':self.model.state_dict(),
            },os.path.join(self.run_path,f'{epoch_index+1}.pth'))


            if epoch_index>0:
                # for model in self.models.keys():
                mpath=os.path.join(self.run_path,'{}.pth'.format(epoch_index))
                if os.path.exists(mpath):
                    os.remove(mpath)      

            if val_loss<self.min_val_loss:
                self.min_val_loss=val_loss
                # for model in self.models.keys():
                torch.save({
                    'lr_scheduler':self.lr_scheduler.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'model':self.model.state_dict(),
                },os.path.join(self.run_path,f'best.pth'))

            info={'starting_epoch':epoch_index+1,'min_val_loss':self.min_val_loss}
            with open(os.path.join(self.run_path,'info.json'), 'w') as f:
                json.dump(info, f)
            



    def _patch_loader(self,dataset,patch_size,queue_length,samples_per_volume,num_workers,batch_size):
        sampler = tio.data.UniformSampler(patch_size)
        patches_queue = tio.Queue(
            dataset,
            queue_length,
            samples_per_volume,
            sampler,
            num_workers=num_workers,
            verbose=False,
        )
        patches_loader = DataLoader(
            patches_queue,
            batch_size=batch_size,
            num_workers=0,  # this must be 0
        )
        logging.info(patches_queue.get_max_memory_pretty())  # log
        return patches_loader