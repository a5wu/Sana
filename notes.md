troubleshooting:
- uncomment the pip install torch in environment_setup.sh
- make sure accelerate/huggingface-hub are stilled in the conda env not in usr/local
- change np in train.sh to 1


multi-scale:
diffusion/data/datasets/sana_data_multi_scale.py
diffusion/model/nets/sana_multi_scale.py

non-multi-scale:
diffusion/data/datasets/sana_data.py




command:
bash train_scripts/train.sh configs/custom_config/Sana_100M_img128.yaml --data.data_dir="[~/train]" --data.type=SanaWebDataset --model.multi_scale=false -- train.train_batch_size=32