import os

import torch
import torch.nn as nn
from torch import optim
from pytorch_model.pairwise.contrastive_loss import ContrastiveLoss
from pytorch_model.pairwise.data_generate import get_generate_pairwise
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from pytorch_model.pairwise.eval_pairwise import eval_pairwise
import torchvision.transforms as transforms

def train_pairwise(model,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',\
                   image_size=256,batch_size=16,resume = '',lr=0.001,num_workers=8,checkpoint="checkpoint",\
                   epochs=20,print_every=1000,adj_brightness=1.0, adj_contrast=1.0):

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True

    model = model.to(device)
    criterion = ContrastiveLoss(device=device).to(device)
    # criterion =
    optimizer = optim.Adam(model.parameters(), lr=lr)



    dataloader_train = get_generate_pairwise(train_set,image_size,batch_size,num_workers)
    dataloader_val = get_generate_pairwise(val_set,image_size,batch_size,num_workers)


    text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
    model.train()
    steps = 0
    running_loss_contrastive = 0
    for epoch in range(0, epochs):
        for img0, img1, label in tqdm(dataloader_train):
            steps += 1

            img0, img1, label = img0.to(device), img1.to(device),label.to(device)
            #         img0, img1 , label = img0, img1 , label
            # print(img0.size())
            optimizer.zero_grad()
            output1, output2= model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            running_loss_contrastive +=loss_contrastive.item()

            if steps % print_every == 0:
                eval_pairwise(model,criterion,text_writer,dataloader_val)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss contrastive: {running_loss_contrastive/print_every:.3f}.. "
                     )
                text_writer.write('Epoch %d, Train loss contrastive %.4f, ' % (
                epoch, running_loss_contrastive / print_every, ))
                text_writer.flush()
                steps = 0
                running_loss_contrastive = 0
                model.train()

        torch.save(model.efficient.state_dict(), os.path.join(checkpoint, 'efficient3_%d.pt' % epoch))
    torch.save(model.efficient.state_dict(), os.path.join(checkpoint, 'efficient3_100.pt'))

if __name__ == "__main__":
    train_pairwise()