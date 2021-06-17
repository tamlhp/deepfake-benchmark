python eval.py --val_set /hdd/tam/extend_data/image_split/3dmm/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0

python eval.py --val_set /hdd/tam/extend_data/image_split/deepfake/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0

python eval.py --val_set /hdd/tam/extend_data/image_split/image_swap2d/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0

python eval.py --val_set /hdd/tam/extend_data/image_split/image_swap3d/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0

python eval.py --val_set /hdd/tam/extend_data/image_split/monkey/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0

python eval.py --val_set /hdd/tam/extend_data/image_split/reenact/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0

python eval.py --val_set /hdd/tam/extend_data/image_split/stargan/ --image_size 256 --checkpoint efft0_mydata_checkpoint/   --resume model_fft_24.pt --adj_brightness=1.0 --adj_contrast=1.0  --gpu_id 0  --batch_size 16 --worker 16 efft --type 0
