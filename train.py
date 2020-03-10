import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import torch.nn as nn
import argparse

# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_set', default="data/train/", help='path to train data ')
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None,required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=str, default = '', help='Resume from checkpoint ')
    parser.add_argument('--print_every',type=int, default = 5000, help='Print evaluate info every step train')
    parser.add_argument('--loss',type=str, default = "bce", help='Loss function use')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnext50, resnext ,gan,meso,xception')

    ## torch
    parser_capsule = subparsers.add_parser('capsule', help='Capsule')
    parser_capsule.add_argument("--seed",type=int,required=False,default=0,help="Manual seed")
    parser_capsule.add_argument("--beta1",type=int,required=False,default=0.9,help="Manual seed")
    parser_drn = subparsers.add_parser('drn', help='DRN  ')
    parser_local_nn = subparsers.add_parser('local_nn', help='Local NN ')
    parser_self_attention = subparsers.add_parser('self_attention', help='Self Attention ')

    parser_resnext50 = subparsers.add_parser('resnext50', help='Resnext50 ')
    parser_resnext101 = subparsers.add_parser('resnext101', help='Resnext101 ')
    parser_mnasnet = subparsers.add_parser('mnasnet', help='mnasnet pytorch ')
    parser_xception_torch = subparsers.add_parser('xception_torch', help='Xception pytorch ')
    parser_xception2_torch = subparsers.add_parser('xception2_torch', help='Xception2 pytorch ')
    parser_dsp_fwa = subparsers.add_parser('dsp_fwa', help='DSP_SWA pytorch ')
    parser_siamese_torch = subparsers.add_parser('siamese_torch', help='Siamese pytorch ')
    parser_siamese_torch.add_argument("--length_embed",type=int,required=False,default=1024,help="Length of embed vector")

    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')

    parser_meso = subparsers.add_parser('meso4', help='Mesonet4')
    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')

    ## tf
    parser_xception_tf = subparsers.add_parser('xception_tf', help='Xceptionnet tensorflow')
    parser_xception_tf = subparsers.add_parser('siamese_tf', help='siamese tensorflow')


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    if model== "capsule":
        from pytorch_model.train_torch import train_capsule
        train_capsule(train_set = args.train_set,val_set = args.val_set,gpu_id=gpu_id,manualSeed=args.seed,resume=args.resume,beta1=args.beta1, \
                      dropout=0.05,image_size=args.image_size,batch_size=args.batch_size,lr=args.lr, \
                      num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter,)
        pass
    elif model == "drn":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "local_nn":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.local_nn import local_nn
        model = local_nn()
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "self_attention":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.self_attention import self_attention
        model = self_attention()
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "resnext50":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import resnext50
        model = resnext50()
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "resnext101":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import resnext101
        model = resnext101()
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "mnasnet":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import mnasnet
        model = mnasnet()
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "xception_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.xception import xception
        model = xception(pretrained=True)
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "xception2_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.xception import xception2
        model = xception2(pretrained=True)
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "dsp_fwa":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.DSP_FWA.models.classifier import SPPNet
        model = SPPNet(backbone=50, num_class=1)
        criterion = None
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            from pytorch_model.focal_loss import FocalLoss
            criterion = FocalLoss(gamma=2)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "siamese_torch":
        from pytorch_model.train_torch import train_siamese
        from pytorch_model.siamese import SiameseNetworkResnet
        model = SiameseNetworkResnet(length_embed = args.length_embed,pretrained=True)
        train_siamese(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,length_embed = args.length_embed,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every)
        pass
    elif model == "gan":
        from tf_model.train_tf import train_gan
        train_gan(train_set = args.train_set,val_set = args.val_set,training_seed=0,checkpoint=args.checkpoint)
        pass
    elif model == "meso4":
        from tf_model.mesonet.model import Meso4
        from tf_model.train_tf import train_cnn
        model = Meso4().model
        loss = 'binary_crossentropy'
        if args.loss == "bce":
            loss = 'binary_crossentropy'
        elif args.loss == "focal":
            from tf_model.focal_loss import BinaryFocalLoss
            loss = BinaryFocalLoss(gamma=2)
        train_cnn(model,loss=loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter)
        pass
    elif model == "xception_tf":
        from tf_model.train_tf import train_cnn
        from tf_model.model_cnn_keras import xception
        model = xception()
        loss = 'binary_crossentropy'
        if args.loss == "bce":
            loss = 'binary_crossentropy'
        elif args.loss == "focal":
            from tf_model.focal_loss import BinaryFocalLoss
            loss = BinaryFocalLoss(gamma=2)
        train_cnn(model,loss=loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batchSize,num_workers=1,checkpoint=args.checkpoint,epochs=args.niter)
        pass
    elif model == "siamese_tf":
        from tf_model.siamese import get_siamese_model
        from tf_model.train_tf import train_siamese
        model = get_siamese_model((256, 256, 3))
        loss = 'binary_crossentropy'
        train_siamese(model,loss = loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter)