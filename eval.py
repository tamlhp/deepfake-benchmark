import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

import argparse
from tf_model.focal_loss import BinaryFocalLoss

# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None,required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=str, default = "0", help='GPU id ')
    parser.add_argument('--resume',type=str, default = "", help='Resume from checkpoint ')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnext50, resnext ,gan,meso,xception')

    ## torch
    parser_capsule = subparsers.add_parser('capsule', help='Capsule')
    parser_drn = subparsers.add_parser('drn', help='DRN  ')
    parser_drn = subparsers.add_parser('local_nn', help='Local NN ')
    parser_drn = subparsers.add_parser('self_attention', help='Self Attention ')

    parser_resnet = subparsers.add_parser('resnext50', help='Resnext50 ')
    parser_resnet = subparsers.add_parser('resnext101', help='Resnext101 ')
    parser_resnet = subparsers.add_parser('mnasnet', help='mnasnet pytorch ')
    parser_resnet = subparsers.add_parser('xception_torch', help='Xception pytorch ')

    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')

    parser_meso = subparsers.add_parser('meso', help='Mesonet')
    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')

    ## tf
    parser_xception_tf = subparsers.add_parser('xception_tf', help='Xceptionnet')


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if model== "capsule":
        from pytorch_model.eval_torch import eval_capsule
        eval_capsule(val_set = args.val_set,resume=args.resume, \
                      image_size=args.image_size,batch_size=args.batch_size, \
                      num_workers=args.workers,checkpoint=args.checkpoint)
        pass
    elif model == "drn":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        eval_cnn(model,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint)
        pass
    elif model == "local_nn":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.loacal_nn import local_nn
        model = local_nn()
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint)
        pass
    elif model == "self_attention":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.self_attention import self_attention
        model = self_attention()
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint)
        pass
    elif model == "resnext50":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.model_cnn_pytorch import resnext50
        model = resnext50()
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint)
        pass
    elif model == "resnext101":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.model_cnn_pytorch import resnext101
        model = resnext101()
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint)
        pass
    elif model == "mnasnet":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.model_cnn_pytorch import mnasnet
        model = mnasnet()
        eval_cnn(model,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint)
        pass
    elif model == "xception_torch":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.xception import xception
        model = xception()
        eval_cnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint)
        pass
    elif model == "gan":
        pass
    elif model == "meso":
        from tf_model.mesonet.model import Meso4
        from tf_model.eval_tf import eval_cnn
        model = Meso4().model
        model.load_weights(args.checkpoint + args.resume)

        eval_cnn(model,val_set = args.val_set,image_size=args.image_size, \
                  batch_size=args.batch_size)
        pass
    elif model == "xception_tf":
        from tf_model.eval_tf import eval_cnn
        from tf_model.model_cnn_keras import xception
        model = xception()
        model.load_weights(args.checkpoint + args.resume)

        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, \
                 batch_size=args.batch_size)
        pass