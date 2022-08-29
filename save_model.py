import os
import torch


def save_model(name, model, optimizer, current_epoch, pre_epoch):
    if pre_epoch != -1:
        pre_path = os.path.join(os.getcwd(), "save", name, "checkpoint_{}.tar".format(pre_epoch))
        os.remove(pre_path)
    cur_path = os.path.join(os.getcwd(), "save", name, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, cur_path)