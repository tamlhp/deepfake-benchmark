# dfd_benchmark
Deep Fake Detection Benchmark

# Enviroment
` pip install -r requirement.txt` 


# Preprocess data
Extract fame from video and detect face in frame to save *.jpg image.

`python extrac_face.py --inp in/ --output out/ --worker 1 --duration 4`

`--inp` : folder contain video

`--output` : folder output .jpg image 

`--worker`  : number thread extract

`--duration` : number of frame skip each extract time

Preprocess for GAN-fingerprint

`python data_preparation_gan.py in_dir /hdd/tam/df_in_the_wild/image/train --out_dir /hdd/tam/df_in_the_wild/gan/train resolution 128`

Preprocess for visual model

`python -m feature_model.visual_artifact.process_data --input_real /hdd/tam/df_in_the_wild/image/train/0_real --input_fake /hdd/tam/df_in_the_wild/image/train/1_df --output /hdd/tam/df_in_the_wild/train_visual.pkl --number_iter 1000`

Preprocess for headpose model

`python -m feature_model.headpose_forensic.process_data --input_real /hdd/tam/df_in_the_wild/image/train/0_real --input_fake /hdd/tam/df_in_the_wild/image/train/1_df --output /hdd/tam/df_in_the_wild/train_visual.pkl --number_iter 1000`

Preprocess for spectrum 

`python -m feature_model.spectrum.process_data --input_real /hdd/tam/df_in_the_wild/image/train/0_real --input_fake /hdd/tam/df_in_the_wild/image/train/1_df --output /hdd/tam/df_in_the_wild/train_spectrum.pkl --number_iter 1000`


#  Train

Train for cnn 

`python train.py --train_set data/Celeb-DF/image/train/ --val_set data/Celeb-DF/image/test/ --batch_size 32 --image_size 128 --workers 16 --checkpoint xception_128_df_inthewild_checkpoint/ --gpu_id 0 --resume model_pytorch_1.pt --print_every 10000000 xception_torch`

Train for feature model

`python train.py --train_set /hdd/tam/df_in_the_wild/train_visual.pkl   --checkpoint spectrum_128_df_inthewild_checkpoint/ --gpu_id 0 --resume model_pytorch_1.pt spectrum`


# Eval

Eval for cnn

`python eval.py  --val_set /hdd/tam/df_in_the_wild/image/test/ --adj_brightness 1.0 --adj_contrast 1.0 --batch_size 32 --image_size 128 --workers 16 --checkpoint efficientdual_128_df_inthewild_checkpoint/ --resume model_dualpytorch3_1.pt efficientdual`

`python eval.py  --val_set /hdd/tam/df_in_the_wild/image/test/ --adj_brightness 1.0 --adj_contrast 1.5 --batch_size 32 --image_size 128 --workers 16 --checkpoint capsule_128_df_inthewild_checkpoint/ --resume 4 capsule`

``

Eval for feature model

`python eval.py --val_set ../DeepFakeDetection/Experiments_DeepFakeDetection/test_dfinthewild.pkl   --checkpoint ../DeepFakeDetection/Experiments_DeepFakeDetection/model_df_inthewild.pkl --resume model_df_inthewild.pkl spectrum`

# Detect

`python detect_img.py --img_path /hdd/tam/extend_data/image/test/1_df/reference_0_113.jpg --model_path efficientdual_mydata_checkpoint/model_dualpytorch3_1.pt --gpu_id 0 efficientdual`

`python detect_img.py --img_path /hdd/tam/extend_data/image/test/1_df/reference_0_113.jpg --model_path xception_mydata_checkpoint/model_pytorch_0.pt --gpu_id 0 xception_torch`

`python detect_img.py --img_path /hdd/tam/extend_data/image/test/1_df/reference_0_113.jpg --model_path capsule_mydata_checkpoint/capsule_1.pt  --gpu_id 0 capsule`

# References
[1] https://github.com/nii-yamagishilab/Capsule-Forensics-v2

[2] Nguyen, H. H., Yamagishi, J., & Echizen, I. (2019). Capsule-forensics: Using Capsule Networks to Detect Forged Images and Videos. ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings, 2019-May, 2307–2311.

[3] https://github.com/PeterWang512/FALdetector

[4] Wang, S.-Y., Wang, O., Owens, A., Zhang, R., & Efros, A. A. (2019). Detecting Photoshopped Faces by Scripting Photoshop.

[5] Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. 

[6] Hsu, C.-C., Zhuang, Y.-X., & Lee, C.-Y. (2020). Deep Fake Image Detection Based on Pairwise Learning. Applied Sciences, 10(1), 370. 

[7] Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2019). MesoNet: A compact facial video forgery detection network. 10th IEEE International Workshop on Information Forensics and Security, WIFS 2018. 

[8] https://github.com/DariusAf/MesoNet

[9] Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2019). Celeb-DF: A New Dataset for DeepFake Forensics.

[10] https://github.com/deepfakeinthewild/deepfake_in_the_wild

[11] https://www.idiap.ch/dataset/deepfaketimit

[12] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF (v2): A new
dataset for deepfake forensics,” arXiv preprint arXiv:1909.12962v3, 2018.

[13] Neves, J. C., Tolosana, R., Vera-Rodriguez, R., Lopes, V., & Proença, H. (2019). Real or Fake? Spoofing State-Of-The-Art Face Synthesis Detection Systems. 13(9), 1–8.

[14] https://github.com/danmohaha/DSP-FWA

