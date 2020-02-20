import os
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_set', default="data/train/", help='path to train data ')
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batchSize', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=1, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = None, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=int, default = 0, help='Resume from checkpoint ')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,gan,meso,xception')

    parser_capsule = subparsers.add_parser('capsule', help='Capsule ')

    parser_gan = subparsers.add_parser('gan', help='GAN fingerprint')

    parser_meso = subparsers.add_parser('meso', help='Mesonet')
    # parser_afd.add_argument('--depth',type=int,default=10, help='AFD depth linit')
    # parser_afd.add_argument('--min',type=float,default=0.1, help='minimum_support')
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)



    data_path = args.data_path
    model = args.model
    if model== "capsule":
        pass
    elif model == "drn":
        pass
    elif model == "gan":
        pass
    elif model == "meso":
        pass
    elif model == "xception":
        pass