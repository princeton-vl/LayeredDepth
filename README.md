# Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation

We introduce **LayeredDepth**, a real and a synthetic dataset tailored to the multi-layer depth estimation task. The real dataset is for benchmark purposes, containing in-the-wild images with high-quality, human-annotated relative depth ground-truth. Complementary to the real-world benchmark, our synthetic dataset allows us to train good-performing models for multi-layer depth estimation.

If you find LayeredDepth useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2503.11633">
        Seeing and Seeing Through the Glass: <br> Real and Synthetic Data for Multi-Layer Depth Estimation
    </a>
</h3>
<p align="center">
    <a href="https://hermera.github.io">Hongyu Wen</a>, 
    <a href="https://zuoym15.github.io">Yiming Zuo</a>, 
    <a href="http://venkatsubra.com/">Venkat Subramanian</a>, 
    <a href="https://patrickchen.me/">Patrick Chen</a>, 
    <a href="https://www.cs.princeton.edu/~jiadeng/">Jia Deng</a><br/>
</p>

```
@article{wen2025layereddepth,
      title={Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation}, 
      author={Hongyu Wen and Yiming Zuo and Venkat Subramanian and Patrick Chen and Jia Deng},
      journal={arXiv preprint arXiv:2503.11633},
      year={2025},
}
```

## Installation
```
conda env create -f env.yaml
conda activate layereddepth
```

## LayeredDepth Benchmark
<img src="imgs/bench_gallery.jpg" width='1000'>

### Download
The benchmark data is available under CC0 license. Download the validation set (images + ground-truth) and test set (images) on [huggingface](https://huggingface.co/datasets/princeton-vl/LayeredDepth).

### Evaluation on Validation Set
Run
```
python3 evaluate.py --subset layer_all # for multi-layer depth estimation
python3 evaluate.py --subset layer_1 # for first layer depth estimation
```

### Evaluation on Test Set
To evaluate your model on the test set and compare your results with the baseline, you need to submit your flow predictions to the [evaluation server](https://layereddepth.cs.princeton.edu).

For each image `i.png` in LayeredDepth (where $i = 0, \dots, 1499$), save your depth estimation for layer $j$ as a 16-bit PNG file named `i_j.png` or numpy file named `i_j.npy` in the estimations directory. For example, the first layer depth estimation for image `0.png` should be named as `0_1.png` or `0_1.npy`. For multi-layer depth estimation, you will need to submit layer 1, 3, 5, 7 depth estimations for each image.

Submit your predictions to the evaluation server using the command below. Ensure your submission follows the same depth estimation format described above. Replace the placeholders with your actual email, submission path, and method name:
```
python3 upload_submission.py --email your_email --path path_to_your_submission --method_name your_method_name --benchmark multi_layer
python3 upload_submission.py --email your_email --path path_to_your_submission --method_name your_method_name --benchmark first_layer
```

Upon submission, you will receive a unique submission ID, which serves as the identifier for your submission. Results are typically emailed within 1 hour. Please note that each email user may upload only three submissions every seven days.

To make your submission public, run the command below. Please replace the placeholders with your specific details, including your submission ID, email, and method name. You may specify the publication name, or use "Anonymous" if the publication is under submission. It's optional to provide URLs for the publication and code.
```
python3 modify_submission.py --id submission_id --email your_email --anonymous False --method_name your_method_name --publication "your publication name" --url_publication "https://your_publication" --url_code "https://your_code"
```

## Synthetic Data Generator

### Set up Blender
To generate multiple layer ground truth from blender, you will need to build customized blender.

First, clone the [Blender](https://github.com/blender/blender) source code to your machine. Using version 4.2.0.
```
git clone --depth 1 --branch v4.2.0 git@github.com:blender/blender.git
```

Next, apply the [multi-layer ground truth patch](multi-layer-ground-truth.patch).
```
git apply multiple-layer-ground-truth.patch 
```
Then build blender from source following [instructions](https://developer.blender.org/docs/handbook/building_blender).

### Infinigen Scene Generator
Our synthetic data generator is based on [Infinigen](https://infinigen.org). We provide the source code tailored for transparent object generation in the `infinigen` directory.

To generate data on your own, please follow instructions for Infinigen data generation [here](infinigen/docs/HelloRoom.md). 

The pre-generated data is available on [huggingface](https://huggingface.co/datasets/princeton-vl/LayeredDepth-Syn) under the BSD 3-Clause license.