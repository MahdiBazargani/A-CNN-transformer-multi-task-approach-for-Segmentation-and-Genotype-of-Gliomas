import torch.nn as nn
import torch


def Dice(output,target,weight=None, eps=1e-5):
    if weight is None:
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
    return num/den
def Dice_loss(output,target,weight=None, eps=1e-5):
    if weight is None:
        num = 2 * (output * target).sum()+1
        den = output.sum()**2 + target.sum()**2 + 1
    return  - num / den
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def Dice(self,output,targets):
        dice=Dice(torch.where(torch.sigmoid(output[:,0])>0.5,1,0), torch.where(targets[:,0]>0,1,0))
        dice+=Dice(torch.where(torch.sigmoid(output[:,1])>0.5,1,0), torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0))
        dice+=Dice(torch.where(torch.sigmoid(output[:,2])>0.5,1,0), torch.where(targets[:,0]==3 ,1,0))
        dice/=3
        return dice
    def _pool(self,data):
        return torch.where(self.pool(data.float())>0.5,1.,0.)
    
    def cross_entropy(self,output,target):
        output= output.unsqueeze(1)
        output = torch.cat([1-output,output],dim=1)
        return self.cross_loss(output,target.long())
    def forward(self, output, targets,n_down):
        self.cross_loss = nn.CrossEntropyLoss()
        self.pool= nn.AvgPool3d(kernel_size=n_down)
        WT = self._pool(torch.where(targets[:,0]>0,1,0)) 
        TC = self._pool(torch.where( torch.logical_or(targets[:,0]==1,targets[:,0]==3) ,1,0))
        ET = self._pool(torch.where(targets[:,0]==3 ,1,0))
        loss=Dice_loss(torch.sigmoid(output[:,0]), WT)
        loss+=Dice_loss(torch.sigmoid(output[:,1]), TC)
        loss+=Dice_loss(torch.sigmoid(output[:,2]), ET)

        loss += torch.nn.functional.binary_cross_entropy_with_logits(output[:,0], WT)
        loss += torch.nn.functional.binary_cross_entropy_with_logits(output[:,1], TC)
        loss += torch.nn.functional.binary_cross_entropy_with_logits(output[:,2], ET)

        #loss +=self.cross_entropy(output[:,0],WT)
        #loss +=self.cross_entropy(output[:,1],TC)
        #loss +=self.cross_entropy(output[:,2],ET)

        loss/=6
        return loss
