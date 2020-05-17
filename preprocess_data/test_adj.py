import torchvision
from PIL import Image
img_path = "../../../extract_raw_img/real/aabqyygbaa.mp4_0.jpg"
im = Image.open(img_path)
im_adj = torchvision.transforms.functional.adjust_contrast(im,4)
im_adj.show()