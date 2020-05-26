### HeadPose Forensics

This repository is the implementation of the work used in our ICASSP paper
[Exposing Deep Fakes Using Inconsistent Head Poses](https://arxiv.org/abs/1811.00661)



#### Environment

- Ubuntu 16.04
- tqdm 4.28.1
- numpy 1.15.4
- dlib 19.16.0
- opencv-python 3.4.3.18

#### Test

```bash
python run_test.py --input_dir=debug_data --classifier_path=path/to/trained/model --save_file=path/to/output/results
```

This will examine all images and videos in the folder of 'debug_data', print results in terminal, and the probability of being fake images/video could be saved in --save_file in the project root folder. 

The result include a optout attribute, which indicate whether a image is not used for classification when it is true.

#### Train

There are 3 steps to train the classifier:

step 1: extract landmarks of real and fake data

```bash
python train_step1_landmarks.py --real_video_dir=dir/to/real/videos --fake_video_dir=dir/to/fake/videos --output_landmark_path=path/to/save/landmarks
```

step 2: extract head poses

```bash
python train_step2_headposes.py --landmark_info_path=path/to/landmarks/in/step1 --headpose_save_path=path/to/save/headpose/data
```

step 3: train svm model

```bash
python train_step3_training.py --headpose_path=path/to/headposes/in/step2 --model_save_path=path/to/save/trained/model
```


#### Citation

Please cite our paper in your publications if it helps your research.
```commandline
@inproceedings{yang2019exposing,
  title={Exposing Deep Fakes Using Inconsistent Head Poses},
  author={Yang, Xin and Li, Yuezun and Lyu, Siwei},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2019}
}
```

#### Notice
This repository is NOT for commecial use. It is provided "as it is" and we are not responsible for any subsequence of using this code.
