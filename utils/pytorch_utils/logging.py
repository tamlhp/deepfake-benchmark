import torch
from torch.utils.tensorboard import SummaryWriter



# writer = SummaryWriter()

class Logger():
    def __init__(self,log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
    def write_scalar(self,scalar_dict: dict,global_step: int):
        for k in scalar_dict:
            self.writer.add_scalar(k, scalar_dict[k], global_step)
    def write_image(self,image_dict: dict,global_step: int):
        for k in image_dict:
            self.writer.add_image(k, image_dict[k], global_step)