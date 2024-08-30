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

# 🛠️Setup & Installation
Our experiment developed and ran with CUDA 12.3 and PyTorch version 2.1.0 and mmseg 1.2.2 version.
- Install every Required Modules : 
```
pip install -r requirements.txt
```
- Place the model into : `adv_manhole/models/monodepth2/weights/{model_name}`
- Get Monodepth2 Model [GitHub](https://github.com/nianticlabs/monodepth2)
- Our CARLA Dataset [Huggingface](https://huggingface.co/datasets/naufalso/carla_hd)

# 💨Training
```
python generate_patch.py --config_path path_to_config_file.yml
```
# 📈Evaluation
For evaluation setup, specify the desired image inside Config file.
Example how to insert specific generated adversarial manhole texture patch for evaluation : 
```
manhole_set:
      adversarial_sample_image:
            normal: path_to_image/manhole.jpg
            trained_patch: path_to_image/trained_patch.png
```
The pipeline can handle multiple images and then calculate the average metrics calculation result from specified sample images.
After specify the desired patch image, we can run the evaluation : 
```
python evaluate_patch.py --config_path path_to_config_file.yml
```
