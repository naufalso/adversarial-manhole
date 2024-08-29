## Adversarial Mahole

This is a PyTorch implementation of the [Adversarial Manhole paper](https://www.arxiv.org/abs/2408.14879): 
```
@misc{suryanto2024adversarialmanholechallengingmonocular,
      title={Adversarial Manhole: Challenging Monocular Depth Estimation and Semantic Segmentation Models with Patch Attack}, 
      author={Naufal Suryanto and Andro Aprila Adiputra and Ahmada Yusril Kadiptya and Yongsu Kim and Howon Kim},
      year={2024},
      eprint={2408.14879},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.14879}, 
}
```

# Setup & Installation
```
pip install -r requirements.txt
```
Get Monodepth2 Model :
Place the model into : `adv_manhole/models/monodepth2/weights/{model_name}`
```
https://github.com/nianticlabs/monodepth2
```

# Training
```
python generate_patch.py --config_path path_to_config_file.yml
```
# Evaluation
```
python evaluate_patch.py --config_path path_to_config_file.yml
```
