import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn 
from torch import optim
import os
from resnet import ResNet18
 

def main():
    os.chdir('/Users/jjvirgo/pytorch/cifar')
    batchsz=64
    cifar_train = datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)

    cifar_test = datasets.CIFAR10('cifar',train=False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_train_loader = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)
    cifar_test_loader = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    model = ResNet18()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    print(model)
    for epoch in range(10):
        model.train()
        for batch_idx,(x,label) in enumerate(cifar_train_loader):
            logits=model(x)
            loss=criteon(logits,label)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(epoch,loss.item())

        model.eval()
        with torch.no_grad():
        #test
            total_correct=0
            total_num=0
            for (x,label) in cifar_test_loader:
                #[b,10]
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct+=torch.eq(pred,label).float().sum()

                total_num+=x.size(0)
            acc= total_correct/total_num
            print(epoch,acc)

if __name__ == '__main__':

    main()