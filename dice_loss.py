import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        smooth = 1.0
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1), target.view(-1)) + 0.0001
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = (2*self.inter.float() + smooth)/(self.union.float() + smooth)
        
        return t

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i+1)


class DiceLoss(_Loss):
    def forward(self, input, target):
        s = 0
        input = torch.sigmoid(input)
#         input = F.softmax(input, dim=1)
        for i in range(input.size(0)):
            s = s + 1 - dice_coeff(input[i], target[i])
        return s / (i+1)
    
class DiceScore(_Loss):
    def forward(self, input, target):
        s = 0
        input = torch.sigmoid(input)
        input[input>0.5]=1.0
        input[input<0.5]=0.0
#         input = F.softmax(input, dim=1)
        for i in range(input.size(0)):
            s = s + dice_coeff(input[i], target[i])
        return s / (i+1)