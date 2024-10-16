import torch
from torch import nn 
from torch.nn import functional as F


class VGG(nn.Module):

    def __init__(self):
        super(VGG,self).__init__()
        
        self.conv_unit = nn.Sequential(
            #[64,3,32,32]> [64,6...]
            nn.Conv3d(1,64,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.MaxPool3d(kernel_size=2,stride=2,padding=1),
            nn.Conv3d(64,128,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.MaxPool3d(kernel_size=2,stride=2,padding=1),
            nn.Conv3d(128,256,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.Conv3d(256,256,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.Conv3d(256,256,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.MaxPool3d(kernel_size=2,stride=2,padding=1),
            nn.Conv3d(256,512,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.Conv3d(512,512,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.Conv3d(512,512,kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.MaxPool3d(kernel_size=2,stride=2,padding=1),
            
            )
            #[64,6...] > [64,16...]
    
        #打平
        self.fc_unit=nn.Sequential(
            nn.Linear(512*5*5*3,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,1000),
            nn.ReLU(),
            nn.Linear(1000,347)
        )

        

        #self.criteon=nn.CrossEntropyLoss()

    
    
    def forward(self,x):
        batchsz=x.size(0)
        #[b,3,32,32]--[b,16,6,6]
        x=self.conv_unit(x)
        #[b,16,6,6]--[b,16*6*6]
        
        x=x.view(batchsz,512*3*5*5)
        #[b,16*6*6]--[b,10]
        logits=self.fc_unit(x)
        #self.criteon=nn.CrossEntropyLoss()
        logits = F.softmax(logits,dim=1)
        #loss=self.criteon(logits,y)
        
        return logits
        



def main():
    net=VGG()
    x=torch.randn(1,3,26,52,52)
    out=net(x)
    print('lenet5:',out.shape)
      

if __name__ == '__main__':
    main()