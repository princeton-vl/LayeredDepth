# Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation

We introduce **LayeredDepth**, a real and a synthetic dataset tailored to the multi-layer depth estimation task. The real dataset is for benchmark purposes, containing in-the-wild images with high-quality, human-annotated relative depth ground-truth. Complementary to the real-world benchmark, our synthetic dataset allows us to train good-performing models for multi-layer depth estimation.

If you find LayeredDepth useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2409.05688">
        Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation
    </a>
</h3>
<p align="center">
    <a href="https://hermera.github.io">Hongyu Wen</a>, 
    <a href="">Yiming Zuo</a>, 
    <a href="">Venkat Subramanian</a>, 
    <a href="">Patrick Chen</a>, 
    <a href="https://www.cs.princeton.edu/~jiadeng/">Jia Deng</a><br/>
</p>

```
Bibtex Placeholder
```

## Installation
```
conda env create -f env.yaml
conda activate layereddepth
```

## LayeredDepth Benchmark
<img src="imgs/bench_gallery.jpg" width='1000'>

### Download
Download the validation set (images + ground-truth) and test set (images) [here](https://drive.google.com/drive/folders/1Vw9BSeXoF2SRiNO199ff-Fa6yAZRwdFM?usp=sharing).

### Evaluation on Validation Set
Unzip the validation set into the data/ directory.
For each image `i.png` in LayeredDepth (where $i = 0, \dots, 1499$), save your depth estimation for layer $j$ as a 16-bit PNG file named `i_j.png` in the estimations directory.

Then run
```
python3 evaluate_all.py # for all relative depth tuples
python3 evaluate_layer1.py # for first layer relative depth tuples
```

### Evaluation on Test Set
To evaluate your model on the test set and compare your results with the baseline, you need to submit your flow predictions to the [evaluation server](https://layereddepth.cs.princeton.edu).

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
Coming Soon!

## Acknowledgements
TBD