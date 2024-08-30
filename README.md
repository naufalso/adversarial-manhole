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

# 🧾 Configs
Config file can be placed inside `./configs` directory. Config files handles dataset, device, training parameter and etc.

Below is how to setup the config files
1. Device & dataset
```
device:
      gpu: gpu_num

dataset:
      name: dataset_name
      cache_dir: cache_directory
      batch_size: 8
```
For now, our supported data loading uses Huggingface dataset. For, CPU usage we can alter the gpu_num into cpu.

2. Manhole Sample
```
manhole_set:
      manhole_candidate_path: directory_to/sample_manhole/
      image_size: 256
      adversarial_sample_images:
            normal: path_to_image/image_normal_manhole.jpg
            artistic: path_to_image/image_artistic_manhole.jpg
            adversarial_manhole: path_to_image/image_trained_manhole.jpg
            . . .
```
`manhole_candidate_path` and `image_size` are mandatory, while `adversarial_sample_images` at least have 1 image to be filled and can be added as many as possible for evaluation purpose.

3. Model, Patches, and Training Parameter
```
model:
      mde_model: mono_640x192
      ss_model: ddrnet_23

patches:
      texture:
            texture_resolution: 256
            texture_augmentation:
                  brightness: 0.2
                  contrast: 0.1
            output_augmentation:
                  brightness: 0.2
                  contrast: 0.1
      loss_init_weight:
            mde_loss_weight: 2.0
            ss_ua_loss_weight: 0.5
            ss_ta_loss_weight: 0.5
            tv_loss_weight: 1.0
            content_loss_weight: 0.5
            background_loss_weight: 0.0

parameter:
      epochs: 25
      learning_rate: 0.01
```
Define Mono Depth Estimation and Semantic Segmentation model that will be attacked. Specify Patches parameter and patches optimization parameter, above also an example of default parameter used in the paper.

4. Log
```
log:
      log_main_dir: log
      log_name: test
      log_prediction: 5
```
Set log details such as log interval and directory to save training/evaluation log.

# 💨Training
After Setup the environment and the config file, we can optimize the adversarial manhole using below command by specifying config file :
```
python generate_patch.py --config_path path_to_config_file.yml
```
The training loss and update of every optimized image will be saved in specified log directory.

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
