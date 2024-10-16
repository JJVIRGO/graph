import torch
from torch import nn 
from torch.nn import functional as F


class VGG(nn.Module):

    def __init__(self):
        super(VGG,self).__init__()
        
        self.conv_unit = nn.Sequential(
            #[64,3,32,32]> [64,6...]
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            )
            #[64,6...] > [64,16...]
    
        #打平
        self.fc_unit=nn.Sequential(
            nn.Linear(512*8*8,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,10),
            nn.ReLU()
        )

        

        #self.criteon=nn.CrossEntropyLoss()

    
    
    def forward(self,x):
        batchsz=x.size(0)
        #[b,3,32,32]--[b,16,6,6]
        x=self.conv_unit(x)
        #[b,16,6,6]--[b,16*6*6]

        x=x.view(batchsz,512*8*8)
        #[b,16*6*6]--[b,10]
        logits=self.fc_unit(x)
        #self.criteon=nn.CrossEntropyLoss()
        #loss=self.criteon(logits,y)
        #logits=F.sigmoid(logits)
        return logits

        return x




def main():
    net=VGG()
    x=torch.randn(3,1,100,100)
    out=net(x)
    print('lenet5:',out.shape)
      

if __name__ == '__main__':
    main()