# rm -rf /root/data/cls/classifi_data/
# python ./classifi_data_process.py H_efficientnet_b4_666
# python ./train.py H_efficientnet_b4_666 /root/data/cls/classifi_data/cut_train/ 8 64 --epochs 200 --image_size 224 --optimizer adamax
# python ./test.py efficientnet_b4 H_efficientnet_b4_666/acc /root/data/cls/classifi_data/cut_val/ 8 --image_size 224

python ./train.py H_efficientnet_b4_777 /root/data/cls/classifi_data_/cut_train/ 6 64 --epochs 100 --lr 0.001 --image_size 224
