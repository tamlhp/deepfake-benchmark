import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import torch.nn as nn
import torch
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
    parser_myresnext = subparsers.add_parser('myresnext', help='My Resnext ')
    parser_mnasnet = subparsers.add_parser('mnasnet', help='mnasnet pytorch ')
    parser_xception_torch = subparsers.add_parser('xception_torch', help='Xception pytorch ')
    parser_xception2_torch = subparsers.add_parser('xception2_torch', help='Xception2 pytorch ')
    parser_dsp_fwa = subparsers.add_parser('dsp_fwa', help='DSP_SWA pytorch ')
    parser_siamese_torch = subparsers.add_parser('siamese_torch', help='Siamese pytorch ')
    parser_siamese_torch.add_argument("--length_embed",type=int,required=False,default=1024,help="Length of embed vector")
    parser_meso = subparsers.add_parser('meso4_torch', help='Mesonet4')


    parser_pairwise = subparsers.add_parser('pairwise', help='Pairwises pytorch ')
    parser_pairwise.add_argument("--mode",type=int,required=True,default=0,help="0: train siamese net, 1: train classify net ")
    parser_pairwise.add_argument("--pair_path",type=str,required=False,default="pairwise_0.pt",help="Path to pairwise network ")

    parser_pairwise_efficient = subparsers.add_parser('pairwise_efficient', help='Pairwises Efficient pytorch ')
    parser_pairwise_efficient.add_argument("--mode",type=int,required=True,default=0,help="0: train siamese net, 1: train classify net ")
    parser_pairwise_efficient.add_argument("--pair_path",type=str,required=False,default="pairwise_0.pt",help="Path to pairwise network ")


    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')
    parser_gan.add_argument("--total_train_img",type=float,required=False,default=10000,help="Total image in training set")
    parser_gan.add_argument("--total_val_img",type=int,required=False,default=2000,help="Total image in testing set")
    parser_m2tr = subparsers.add_parser('m2tr', help='')
    parser_m2tr.add_argument("--backbone", type=str, default="efficientnet-b0", help="")
    parser_m2tr.add_argument("--texture_layer", type=str, default="b2", help="")
    parser_m2tr.add_argument("--feature_layer", type=str, default="final", help="")
    parser_m2tr.add_argument("--depth", type=int, default=0, help="")
    parser_m2tr.add_argument("--drop_ratio", type=float, default=0.0, help="")
    parser_m2tr.add_argument("--has_decoder", type=int, default=0, help="")

    parser_f3net = subparsers.add_parser('f3net', help='')
    parser_f3net.add_argument("--mode", type=str, default="Both", help="[Original, FAD, LFS, Both, Mix]")

    parser_mat = subparsers.add_parser('mat', help='')
    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')
    parser_efficient = subparsers.add_parser('efficient', help='Efficient Net')
    parser_efficient.add_argument("--type",type=str,required=False,default="0",help="Type efficient net 0-8")
    parser_efficientdual = subparsers.add_parser('efficientdual', help='Efficient Net')
    parser_efft = subparsers.add_parser('efft', help='Efficient Net fft')
    parser_efft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    parser_e4dfft = subparsers.add_parser('e4dfft', help='Efficient Net 4d fft')
    parser_e4dfft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")
    ## tf
    parser_meso = subparsers.add_parser('meso4', help='Mesonet4')
    parser_xception_tf = subparsers.add_parser('xception_tf', help='Xceptionnet tensorflow')
    parser_siamese_tf = subparsers.add_parser('siamese_tf', help='siamese tensorflow')

    ##############  gc
    parser_spectrum = subparsers.add_parser('spectrum', help='siamese tensorflow')
    parser_headpose = subparsers.add_parser('heapose', help='siamese tensorflow')
    parser_visual = subparsers.add_parser('visual', help='siamese tensorflow')

    ## adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')

    return parser.parse_args()

def get_criterion_torch(arg_loss):
    criterion = None
    if arg_loss == "bce":
        criterion = nn.BCELoss()
    elif arg_loss == "focal":
        from pytorch_model.focal_loss import FocalLoss
        criterion = FocalLoss(gamma=2)
    return criterion

def get_loss_tf(arg_loss):
    loss = 'binary_crossentropy'
    if arg_loss == "bce":
        loss = 'binary_crossentropy'
    elif arg_loss == "focal":
        from tf_model.focal_loss import BinaryFocalLoss
        loss = BinaryFocalLoss(gamma=2)
    return loss
if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    adj_brightness = float(args.adj_brightness)
    adj_contrast = float(args.adj_contrast)
    if model== "capsule":
        from pytorch_model.train_torch import train_capsule
        train_capsule(train_set = args.train_set,val_set = args.val_set,gpu_id=gpu_id,manualSeed=args.seed,resume=args.resume,beta1=args.beta1, \
                      dropout=0.05,image_size=args.image_size,batch_size=args.batch_size,lr=args.lr, \
                      num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter,\
                      adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "drn":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "local_nn":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.local_nn import local_nn
        model = local_nn()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "self_attention":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.self_attention import self_attention
        model = self_attention()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "resnext50":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import resnext50
        model = resnext50()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "resnext101":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import resnext101
        model = resnext101()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "myresnext":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import MyResNetX
        model = MyResNetX()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "mnasnet":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import mnasnet
        model = mnasnet()
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "xception_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.xception import xception
        model = xception(pretrained=True)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "xception2_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.xception import xception2
        model = xception2(pretrained=True)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "meso4_torch":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.model_cnn_pytorch import mesonet
        model = mesonet(image_size=args.image_size)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == 'm2tr':
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.m2tr.m2tr import M2TR

        model_cfg = {
            'IMG_SIZE': args.image_size,
            'BACKBONE': args.backbone,
            'TEXTURE_LAYER': args.texture_layer,
            'FEATURE_LAYER': args.feature_layer,
            'DEPTH': args.depth,
            'NUM_CLASSES': 1,
            'DROP_RATIO': args.drop_ratio,
            'HAS_DECODER': args.has_decoder
        }
        model = M2TR(model_cfg)
        args_txt = "batch{}_lr{}".format(args.batch_size, args.lr)
        # args_txt += "_drmlp{}_aug{}".format(args.dropout_in_mlp, args.augmentation)
        criterion = get_criterion_torch(args.loss)

        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
    elif model == 'f3net':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.synchronize()
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.f3net.models import F3Net

        model = F3Net(mode=args.mode, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        args_txt = "f3net_batch{}_lr{}-".format(args.batch_size, args.lr)
        criterion = get_criterion_torch(args.loss)

        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)
    elif model == 'mat':
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.mat.models.MAT import MAT

        model = MAT()
        args_txt = "mat_batch{}_lr{}-".format(args.batch_size, args.lr)
        criterion = get_criterion_torch(args.loss)

        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every, adj_brightness=adj_brightness,
                  adj_contrast=adj_contrast)

    elif model == "dsp_fwa":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.DSP_FWA.models.classifier import SPPNet
        model = SPPNet(backbone=50, num_class=1)
        criterion = get_criterion_torch(args.loss)
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "siamese_torch":
        from pytorch_model.train_torch import train_siamese
        from pytorch_model.siamese import SiameseNetworkResnet
        model = SiameseNetworkResnet(length_embed = args.length_embed,pretrained=True)
        train_siamese(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,length_embed = args.length_embed,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "pairwise":
        from pytorch_model.pairwise.train_pairwise import train_pairwise
        from pytorch_model.pairwise.model import Pairwise,ClassifyFull
        if args.mode == 0:
            model = Pairwise(args.image_size)
            train_pairwise(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                      batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                      epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        else:
            from pytorch_model.train_torch import train_cnn
            import torch
            model = ClassifyFull(args.image_size)
            model.cffn.load_state_dict(torch.load(os.path.join(args.checkpoint, args.pair_path)))
            criterion = get_criterion_torch(args.loss)
            train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                      image_size=args.image_size, resume=args.resume, \
                      batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                      epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "pairwise_efficient":
        from pytorch_model.efficientnet.train_pairwise import train_pairwise
        from pytorch_model.efficientnet.model_pairwise import EfficientPairwise,EfficientFull
        if args.mode == 0:
            model = EfficientPairwise()
            train_pairwise(model,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                      batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                      epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        else:
            from pytorch_model.train_torch import train_cnn
            import torch
            model = EfficientFull()
            model.efficient.load_state_dict(torch.load(os.path.join(args.checkpoint, args.pair_path)))
            criterion = get_criterion_torch(args.loss)
            train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                      image_size=args.image_size, resume=args.resume, \
                      batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                      epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "efficient":
        from pytorch_model.train_torch import train_cnn
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b'+args.type,num_classes=1)
        model = nn.Sequential(model,nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "efficientdual":
        from pytorch_model.train_torch import train_dualcnn
        from pytorch_model.efficientnet import EfficientDual

        model = EfficientDual()
        criterion = get_criterion_torch(args.loss)
        train_dualcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "efft":
        from pytorch_model.train_torch import train_fftcnn
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=1)
        model = nn.Sequential(model, nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_fftcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "e4dfft":
        from pytorch_model.train_torch import train_4dfftcnn
        from pytorch_model.efficientnet import EfficientNet

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=4)
        model = nn.Sequential(model, nn.Sigmoid())
        criterion = get_criterion_torch(args.loss)
        train_4dfftcnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
# ---------------------------------------------------------------------------------------------
    elif model == "gan":
        from tf_model.train_tf import train_gan
        train_gan(train_set = args.train_set,val_set = args.val_set,training_seed=0,\
                  image_size=args.image_size,batch_size=args.batch_size,num_workers=args.workers, \
                  epochs=args.niter,checkpoint=args.checkpoint,total_train_img = args.total_train_img,total_val_img = args.total_val_img, \
                adj_brightness = adj_brightness, adj_contrast = adj_contrast)
        # train_gan()
        pass
    elif model == "meso4":
        from tf_model.mesonet.model import Meso4
        from tf_model.train_tf import train_cnn
        model = Meso4(image_size=args.image_size).model
        loss = get_loss_tf(args.loss)
        train_cnn(model,loss=loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter, \
                  adj_brightness=adj_brightness, adj_contrast=adj_contrast)
        pass
    elif model == "xception_tf":
        from tf_model.train_tf import train_cnn
        from tf_model.model_cnn_keras import xception
        model = xception(image_size=args.image_size)
        loss = get_loss_tf(args.loss)
        train_cnn(model,loss=loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batchSize,num_workers=1,checkpoint=args.checkpoint,epochs=args.niter, \
                  adj_brightness=adj_brightness, adj_contrast=adj_contrast)
        pass
    elif model == "siamese_tf":
        from tf_model.siamese import get_siamese_model
        from tf_model.train_tf import train_siamese
        model = get_siamese_model((args.image_size, args.image_size, 3))
        loss = 'binary_crossentropy'
        train_siamese(model,loss = loss,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter, \
                      adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    ###############
    elif model == "spectrum":
        from feature_model.spectrum.train_spectrum import train_spectrum

        train_spectrum(args.train_set,model_file=args.checkpoint + args.resume)
        pass
    elif model == "headpose":
        from feature_model.headpose_forensic.train_headpose import train_headpose
        train_headpose(args.train_set,model_file=args.checkpoint + args.resume)
        pass
    elif model == "visual":
        from feature_model.visual_artifact.train_visual import train_visual
        train_visual(args.train_set,model_file=args.checkpoint + args.resume)
        pass
