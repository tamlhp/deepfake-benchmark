import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

import argparse
import torch.nn as nn

# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None,required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=str, default = "", help='Resume from checkpoint ')
    parser.add_argument('--time',type=bool, default = False, help='Print time ')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnext50, resnext ,gan,meso,xception')

    ## torch
    parser_capsule = subparsers.add_parser('capsule', help='Capsule')
    parser_drn = subparsers.add_parser('drn', help='DRN  ')
    parser_local_nn = subparsers.add_parser('local_nn', help='Local NN ')
    parser_self_attention = subparsers.add_parser('self_attention', help='Self Attention ')

    parser_resnext50 = subparsers.add_parser('resnext50', help='Resnext50 ')
    parser_resnext101 = subparsers.add_parser('resnext101', help='Resnext101 ')
    parser_mnasnet = subparsers.add_parser('mnasnet', help='mnasnet pytorch ')
    parser_xception_torch = subparsers.add_parser('xception_torch', help='Xception pytorch ')
    parser_xception2_torch = subparsers.add_parser('xception2_torch', help='Xception2 pytorch ')
    parser_pairwise = subparsers.add_parser('pairwise', help='Pairwises pytorch ')

    parser_pairwise = subparsers.add_parser('pairwise_efficient', help='Pairwises Efficient pytorch ')
    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')
    parser_gan.add_argument("--total_val_img",type=int,required=False,default=2000,help="Total image in testing set")

    parser_efficient = subparsers.add_parser('efficient', help='Efficient Net')
    parser_efficient.add_argument("--type",type=str,required=False,default="0",help="Type efficient net 0-8")
    parser_efficientdual = subparsers.add_parser('efficientdual', help='Efficient Net')
    parser_efft = subparsers.add_parser('efft', help='Efficient Net fft')
    parser_efft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    parser_e4dfft = subparsers.add_parser('e4dfft', help='Efficient Net 4d fft')
    parser_e4dfft.add_argument("--type", type=str, required=False, default="0", help="Type efficient net 0-8")

    ## tf
    parser_meso = subparsers.add_parser('meso4', help='Mesonet 4')
    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')


    parser_xception_tf = subparsers.add_parser('xception_tf', help='Xceptionnet')


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >= 0 else -1
    if model== "capsule":
        from pytorch_model.eval_torch import eval_capsule
        eval_capsule(val_set = args.val_set,gpu_id=int(gpu_id),resume=args.resume, \
                      image_size=args.image_size,batch_size=args.batch_size, \
                      num_workers=args.workers,checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "drn":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.drn.drn_seg import DRNSub
        model = DRNSub(1)
        eval_cnn(model,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "local_nn":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.local_nn import local_nn
        model = local_nn()
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "self_attention":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.self_attention import self_attention
        model = self_attention()
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "resnext50":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.model_cnn_pytorch import resnext50
        model = resnext50(pretrained=False)
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "resnext101":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.model_cnn_pytorch import resnext101
        model = resnext101(pretrained=False)
        eval_cnn(model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "mnasnet":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.model_cnn_pytorch import mnasnet
        model = mnasnet(pretrained=False)
        eval_cnn(model,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,num_workers=args.workers,checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "xception_torch":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.xception import xception
        model = xception(pretrained=False)
        eval_cnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "xception2_torch":
        from pytorch_model.eval_torch import eval_cnn
        from pytorch_model.xception import xception2
        model = xception2(pretrained=False)
        eval_cnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint,show_time=args.time)
        pass
    elif model == "gan":
        from tf_model.eval_tf import eval_gan
        eval_gan(val_set=args.val_set,checkpoint=args.checkpoint,total_val_img=args.total_val_img,show_time=args.time)
        pass
    elif model == "pairwise":
        from pytorch_model.pairwise.model import ClassifyFull
        from pytorch_model.eval_torch import eval_cnn
        import torch
        model = ClassifyFull(args.image_size)
        model.cffn.load_state_dict(torch.load(os.path.join(args.checkpoint, args.pair_path)))
        eval_cnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, show_time=args.time)

    elif model == "pairwise_efficient":
        from pytorch_model.efficientnet.model_pairwise import EfficientPairwise,EfficientFull
        from pytorch_model.eval_torch import eval_cnn
        model = EfficientFull()
        eval_cnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, show_time=args.time)
    elif model == "efficient":
        from pytorch_model.efficientnet import EfficientNet
        from pytorch_model.eval_torch import eval_cnn

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1)
        model = nn.Sequential(model, nn.Sigmoid())
        eval_cnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, show_time=args.time)

    elif model == "efficientdual":
        from pytorch_model.efficientnet import EfficientDual
        from pytorch_model.eval_torch import eval_dualcnn

        model = EfficientDual()
        eval_dualcnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, show_time=args.time)
        pass
    elif model == "efft":
        from pytorch_model.efficientnet import EfficientNet
        from pytorch_model.eval_torch import eval_fftcnn

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=1)
        model = nn.Sequential(model, nn.Sigmoid())
        eval_fftcnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, show_time=args.time)
    elif model == "e4dfft":
        from pytorch_model.efficientnet import EfficientNet
        from pytorch_model.eval_torch import eval_4dfftcnn

        model = EfficientNet.from_pretrained('efficientnet-b' + args.type, num_classes=1,in_channels=4)
        model = nn.Sequential(model, nn.Sigmoid())
        eval_4dfftcnn(model=model, val_set=args.val_set, image_size=args.image_size, resume=args.resume, \
                 batch_size=args.batch_size, num_workers=args.workers, checkpoint=args.checkpoint, show_time=args.time)


    # ----------------------------------------------------
    elif model == "meso4":
        from tf_model.mesonet.model import Meso4
        from tf_model.eval_tf import eval_cnn
        model = Meso4(image_size=args.image_size).model
        model.load_weights(args.checkpoint + args.resume)
        loss = 'binary_crossentropy'
        eval_cnn(model,loss=loss,val_set = args.val_set,image_size=args.image_size, \
                  batch_size=args.batch_size)
        pass
    elif model == "xception_tf":
        from tf_model.eval_tf import eval_cnn
        from tf_model.model_cnn_keras import xception
        from tf_model.focal_loss import BinaryFocalLoss

        model = xception(image_size=args.image_size)
        model.load_weights(args.checkpoint + args.resume)
        loss = BinaryFocalLoss(gamma=2)
        eval_cnn(model,loss=loss, val_set=args.val_set, image_size=args.image_size, \
                 batch_size=args.batch_size)
        pass