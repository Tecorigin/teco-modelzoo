#如果是使用自动迁移环境变量迁移，请在test.sh中添加自动迁移环境变量，注意：请自测后提交
python run_resnet.py --model_name resnet50 --nproc_per_node 4 --bs 64 --lr 0.256 --device sdaa --epoch 50 --dataset_path /datasets/imagenet/ --grad_scale True --autocast True
