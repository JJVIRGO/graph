import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn 
from torch import optim
import os
#from Lenet5 import Lenet5
from resnet import ResNet18
from cifartest import VGG
import vggcifar
def initialize_weights_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
def main():
    os.chdir('F:/10x/pytorch')
    device = torch.device("cuda")
    batchsz=64
    cifar_train = datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ]),download=True)

    cifar_test = datasets.CIFAR10('cifar',train=False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_train_loader = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)
    cifar_test_loader = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    model = vggcifar.__dict__["vgg19"]().to(device)
    initialize_weights_kaiming(model)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-3)
    print(model)
    for epoch in range(20):
        model.train()
        for batch_idx,(x,label) in enumerate(cifar_train_loader):
            x, label = x.to(device), label.to(device)
            logits=model(x)
            loss=criteon(logits,label)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(epoch,"LOSS=",loss.item())

        model.eval()
        with torch.no_grad():
        #test
            total_correct=0
            total_num=0
            for (x,label) in cifar_test_loader:
                x, label = x.to(device), label.to(device)
                #[b,10]
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct+=torch.eq(pred,label).float().sum()

                total_num+=x.size(0)
            acc= float(total_correct/total_num)
            print(epoch,"acc=",acc)

if __name__ == '__main__':

    main()
