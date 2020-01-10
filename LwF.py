import torch
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class LwFmodel:

    def __init__(self,numclass,feature_extractor,batch_size,epochs,learning_rate,task_size):
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.task_size=task_size
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                  transforms.RandomCrop((32,32),padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                   transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.train_dataset = iCIFAR100('../dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('../dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size

        self.train_loader=None
        self.test_loader=None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes=[self.numclass-self.task_size,self.numclass]
        self.train_loader,self.test_loader=self._get_train_and_test_dataloader(classes)
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader


    # train model
    # compute loss
    # evaluate model
    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9,nesterov=True, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, momentum=0.9,nesterov=True,weight_decay=0.00001)
                print("change learning rate%.3f" % (self.learning_rate / 5))
            elif epoch == 68:
                opt = optim.SGD(self.model.parameters(), lr=self.learning_rate /25, momentum=0.9,nesterov=True,weight_decay=0.00001)
                print("change learning rate%.5f" % (self.learning_rate / 25))
            elif epoch == 85:
                opt = optim.SGD(self.model.parameters(), lr=self.learning_rate /125,momentum=0.9,nesterov=True, weight_decay=0.00001)
                print("change learning rate%.5f" % (self.learning_rate / 125))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                opt.zero_grad()
                loss=self._compute_loss(images,target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss.item()))
            accuracy = self._test(self.test_loader)
            print('epoch:%d,accuracy:%.5f' % (epoch, accuracy))
        return accuracy

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy


    def _compute_loss(self, imgs, target):
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)


    def afterTrain(self,accuracy):
        self.numclass+=self.task_size
        filename='model/5_increment:%d_net.pkl' % (self.numclass-self.task_size)
        torch.save(self.model,filename)
        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()


    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data
