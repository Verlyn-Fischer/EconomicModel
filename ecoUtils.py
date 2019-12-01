from torch.utils.tensorboard import SummaryWriter

def writeMetrics(totalProduction,frame,experiment):
    writer = SummaryWriter('runs/'+experiment)
    writer.add_scalar(tag='_TotalProduction',scalar_value=totalProduction,global_step=frame)
    writer.close()
