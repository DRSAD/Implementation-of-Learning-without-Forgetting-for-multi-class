from LwF import LwFmodel
from ResNet import resnet18_cbam
parser=10
numclass=int(100/parser)
task_size=int(100/parser)
feature_extractor=resnet18_cbam()
img_size=32
batch_size=128
task_size=int(100/parser)
memory_size=2000
epochs=100
learning_rate=2.0

model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size)

for i in range(10):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
