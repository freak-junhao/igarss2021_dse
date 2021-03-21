# pooling='max' dropout=0.5 decoder_channels=[256, 128, 64]
# python ./train.py H_unet_res183_0 /root/data/seg_rgb/train/ 8 --epochs 150 --image_size 800 --lr 0.0001 --optimizer adamax
# python ./train.py H_unet_res183_1 /root/data/seg_rgb/train/ 8 --epochs 200 --image_size 800 --lr 0.001

# python ./test.py unet H_unet_res183_0/41 /root/data/seg_rgb/val/ --image_size 800
# python ./test.py unet H_unet_res183_0/acc /root/data/seg_rgb/val/ --image_size 800
# python ./test.py unet H_unet_res183_1/101 /root/data/seg_rgb/val/ --image_size 800
# python ./test.py unet H_unet_res183_1/acc /root/data/seg_rgb/val/ --image_size 800

# pooling='max' dropout=0.5 decoder_channels=[256, 128, 64]
# python ./train.py H_uplus_res183_0 /root/data/seg_rgb/train/ 8 --epochs 150 --image_size 800 --lr 0.0001 --optimizer adamax
python ./train.py H_uplus_res183_1 /root/data/seg_rgb/train/ 8 --epochs 200 --image_size 800 --lr 0.001

# pooling='max' dropout=0.5 decoder_channels=[128, 64, 32]
# python ./train.py H_unet_res183_2 /root/data/seg_rgb/train/ 8 --epochs 150 --image_size 800 --lr 0.0001 --optimizer adamax
python ./train.py H_unet_res183_3 /root/data/seg_rgb/train/ 8 --epochs 200 --image_size 800 --lr 0.001

# python ./test.py uplus H_uplus_res183_0/41 /root/data/seg_rgb/val_/ --image_size 800
# python ./test.py uplus H_uplus_res183_0/acc /root/data/seg_rgb/val_/ --image_size 800
# python ./test.py unet H_unet_res183_2/105 /root/data/seg_rgb/val_/ --image_size 800
# python ./test.py unet H_unet_res183_2/acc /root/data/seg_rgb/val_/ --image_size 800

