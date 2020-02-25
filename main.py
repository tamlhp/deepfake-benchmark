import os
import argparse
import random
from tf_model.focal_loss import BinaryFocalLoss

# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_set', default="data/train/", help='path to train data ')
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batchSize', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=int, default = 0, help='Resume from checkpoint ')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnet ,gan,meso,xception')

    parser_capsule = subparsers.add_parser('capsule', help='Capsule')
    parser_capsule.add_argument("seed",type=int,default=0,help="Manual seed")
    parser_drn = subparsers.add_parser('drn', help='Capsule ')

    parser_resnet = subparsers.add_parser('resnet', help='Capsule ')

    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')

    parser_meso = subparsers.add_parser('meso', help='Mesonet')
    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    if model== "capsule":
        from pytorch_model.train import train_capsule
        train_capsule(train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',manualSeed=0,resume=0,beta1=0.9,dropout=0.05,image_size=256,batch_size=16,lr=0.003,num_workers=1,checkpoint="checkpoint",epochs=20,)
        pass
    elif model == "drn":
        from pytorch_model.train import train_cnn
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        train_cnn(model,train_set = args.train_set,val_set =args.val_set,image_size=args.image_size,batch_size=args.batchSize,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter,print_every=5000)
        pass
    elif model == "resnet":
        from pytorch_model.train import train_cnn
        from pytorch_model.model_cnn_pytorch import Resnext50
        model = Resnext50()
        train_cnn(model,train_set = args.train_set,val_set =args.val_set,image_size=args.image_size,batch_size=args.batchSize,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter,print_every=5000)
        pass
    elif model == "gan":
        pass
    elif model == "meso":
        from tf_model.mesonet.model import Meso4
        from tf_model.train import train_cnn
        model = Meso4().model
        loss = 'binary_crossentropy'
        train_cnn(model,loss,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',image_size=256,batch_size=16,num_workers=1,checkpoint="checkpoint",epochs=20)
        pass
    elif model == "xception":
        from tf_model.train import train_cnn
        from tf_model.model_cnn_keras import xception
        model = xception()
        loss = BinaryFocalLoss(gamma=2)
        train_cnn(model,loss,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',image_size=256,batch_size=16,num_workers=1,checkpoint="checkpoint",epochs=20)
        pass