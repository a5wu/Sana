troubleshooting:
- uncomment the pip install torch in environment_setup.sh
- make sure accelerate/huggingface-hub are stilled in the conda env not in usr/local
- change np in train.sh to 1


multi-scale:
diffusion/data/datasets/sana_data_multi_scale.py
model/nets/sana_multi_scale.py

