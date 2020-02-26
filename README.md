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

#  Train

`python train --train_set train/ --val_set val/ --imageSize 256 resnet`