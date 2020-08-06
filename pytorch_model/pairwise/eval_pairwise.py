
import os

import torch
import torch.nn as nn
from torch import optim
from pytorch_model.pairwise.contrastive_loss import ContrastiveLoss
from pytorch_model.pairwise.data_generate import get_generate_pairwise
import torch.backends.cudnn as cudnn
from tqdm import tqdm


def eval_pairwise(model,criterion,text_writer,dataloader_val):

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    test_loss_contrastive = 0
    model.eval()
    with torch.no_grad():
        for img0, img1, label in dataloader_val:
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output1, output2 = model.forward(img0, img1)
            batch_loss_contrastive = criterion(output1, output2, label)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss_contrastive += batch_loss_contrastive.item()


    print( f"Test loss contrastive: {test_loss_contrastive/len(dataloader_val):.3f}.. ")
    text_writer.write('Test loss contrastive %.4f' % (test_loss_contrastive / len(dataloader_val)))
    text_writer.flush()
    model.train()
    return
