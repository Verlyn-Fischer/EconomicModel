from torch.utils.tensorboard import SummaryWriter
import pickle
import os

def writeMetrics(totalProduction,frame,experiment):
    writer = SummaryWriter('runs/'+experiment)
    writer.add_scalar(tag='_TotalProduction',scalar_value=totalProduction,global_step=frame)
    writer.close()

def saveMemory(fileName, memory):
    if len(memory) > 0:
        file = open(fileName, 'wb')
        pickle.dump(memory,file)

def readMemory(fileName):
    if os.path.exists(fileName):
        with open(fileName, 'rb') as fileObj:
            memory = pickle.load(fileObj)
        return memory
    else:
        return []