import os
import torch
import torch.backends.cudnn as cudnn
from pytorch_model.efficientnet import EfficientNet
from pytorch_model.efficientnet.model import Identity

from feature_model.visual_artifact.pipeline.eyecolor import extract_eyecolor_features
from feature_model.visual_artifact.process_data import load_facedetector
from feature_model.visual_artifact.pipeline.face_utils import *
from feature_model.visual_artifact.pipeline import pipeline_utils
from feature_model.visual_artifact.pipeline.texture import extract_features_eyes,extract_features_faceborder,extract_features_mouth,extract_features_nose


def train_conbie(image_dir, checkpoint, output):
    model_efficient = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1)
    model_efficient._dropout = Identity()
    model_efficient._fc = Identity()
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True

    model_efficient = model_efficient.to(device)
    model_efficient.eval()
    # cnn_feature = model_efficient(img)
