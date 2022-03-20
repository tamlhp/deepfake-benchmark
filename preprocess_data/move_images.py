import glob
import os
import shutil
import random
from tqdm import tqdm

random.seed(0)
train_folder = "/home/tampm/df_in_the_wild/image_jpg/train/"
list_image_1_df = glob.glob("/home/tampm/df_in_the_wild/image_jpg/train/1_df/*")
list_image_0_real = glob.glob("/home/tampm/df_in_the_wild/image_jpg/train/0_real/*")
val_folder = train_folder.replace("/train/","/val/")
random.shuffle(list_image_1_df)
random.shuffle(list_image_0_real)

if not os.path.exists(val_folder):
    os.makedirs(val_folder)
if not os.path.exists(val_folder+"/1_df/"):
    os.makedirs(val_folder+"/1_df/")
if not os.path.exists(val_folder+"/0_real/"):
    os.makedirs(val_folder+"/0_real/")
for i in tqdm(range(50000)):
    df_image = list_image_1_df.pop()
    real_image = list_image_0_real.pop()
    shutil.move(df_image,val_folder+"/1_df/")
    shutil.move(real_image,val_folder+"/0_real/")