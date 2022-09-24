# Pluralistic Image Completion with Gaussian Mixture Model (PICMM)

<img src='imgs/intro.png'/>

### [Paper (ArXiv)](https://arxiv.org/pdf/2205.09086.pdf) | Pre-trained Models (coming soon) | Supplemental Material (coming soon)

**This repository is the official pytorch implementation of our Neurips 2022 paper, *Pluralistic Image Completion with Gaussian Mixture Model*.**

Xiaobo Xia<sup>1\*</sup>,
Wenhao Yang<sup>2\*</sup>,
Jie Ren<sup>3</sup>,
Yewen Li<sup>4</sup>,  
Yibing Zhan<sup>5</sup>, 
Bo Han<sup>6</sup>, 
Tongliang Liu<sup>1</sup> <br>
<sup>1</sup>The University of Sydney, <sup>2</sup>Nanjing University, <sup>3</sup>The University of Edinburgh, <sup>4</sup>Nanyang Technological University, <sup>5</sup>JD Explore Academy, <sup>6</sup>Hong Kong Baptist University
\* Equal contributions



## Prerequisites

- Python >=3.7
- NVIDIA GPU + CUDA cuDNN
```bash
pip install -r requirements.txt
```



## Pipeline

<img src='imgs/Pipeline.png'/>



## Training

```
python train.py --name [exp_name] \
                --k [numbers_of_distributions] \
                --img_file [training_image_path]
```

Notes of training process: 
+ Our code supports 



## Inference

```
python test.py --name [exp_name] \
               --k [numbers_of_distributions] \
               --which_iter [which_iterations_to_load] \
               --img_file [training_image_path] \
               --sample_num [number_of_diverse_results_to_sample]
```

Notes of inference: 
+ `--sample_num`: How many completion results do you want?



## More explanations on back-propogate-max-operation loss

As shown in our paper, the back-propogate-max-operation loss is

$$
\begin{aligned}
    L_{\text{BM}}&
    =-\frac{1}{2}\log\frac{|\pmb{\Sigma}_j^{\hat{\mathbf{z}}_c}|}{|\pmb{\Sigma}^{\mathbf{z}_c}|}
    + \frac{1}{2}\text{tr}\left({(\pmb{\Sigma}^{\mathbf{z}_c})}^{-1}\pmb{\Sigma}_j^{\hat{\mathbf{z}}_c}\right)\\
    &-\frac{1}{2}(\pmb{\mu}_j^{\hat{\mathbf{z}}_c} - \mu^{\mathbf{z}_c})^\top{(\pmb{\Sigma}^{\mathbf{z}_c})}^{-1}(\pmb{\mu}_j^{\hat{\mathbf{z}}_c} - \pmb{\mu}^{\mathbf{z}_c}),
\end{aligned}
$$



Note that we have $\pmb{\mu}^{\mathbf{z}_c},\pmb{\mu}_j^{\hat{\mathbf{z}}_c} \in \mathbb{R}^{256}$ and $\pmb{\Sigma}^{\mathbf{z}_c},\pmb{\Sigma}_j^{\hat{\mathbf{z}}_c}\in \mathbb{R}^{256\times 256}$. In our code, `z_c_mu, z_c_mu_hat_j` $\in \mathbb{R}^{256}$ and `z_c_logsigma, z_c_logsigma_hat_j` $\in \mathbb{R}^{256}$ which represent the diagonal elements of  $\pmb{\Sigma}^{\mathbf{z}_c},\pmb{\Sigma}_j^{\hat{\mathbf{z}}_c}$ under our assumptions.

In this way, we can get this form of the code:
$$
\begin{aligned}
-\frac{1}{2}\log\frac{|\pmb{\Sigma}_j^{\hat{\mathbf{z}}_c}|}{|\pmb{\Sigma}^{\mathbf{z}_c}|} &= -\frac{1}{2}\log\frac{\prod\limits_{i=1}^{256}\text{z\_c\_logsigma[i].exp()}}{\prod\limits_{i=1}^{256}\text{z\_c\_logsigma\_hat\_j[i].exp()}} \\
&= - \frac{1}{2} \times \left(\sum\limits_{i=1}^{256}\text{z\_c\_logsigma[i]} - \sum\limits_{i=1}^{256}\text{z\_c\_logsigma\_hat\_j[i]}\right) \\
&= - \frac{1}{2} \times \left(\text{z\_c\_logsigma} - \text{z\_c\_logsigma\_hat\_j}\right)
\end{aligned}
$$
In the second term, 
$$
\begin{aligned}
    \frac{1}{2}\text{tr}\left({(\pmb{\Sigma}^{\mathbf{z}_c})}^{-1}\pmb{\Sigma}_j^{\hat{\mathbf{z}}_c}\right) &= \frac{1}{2} \left(\sum\limits_{i=1}^{256}(1/\text{z\_c\_logsigma[i].exp()})\times \text{z\_c\_logsigma\_hat\_j.exp()}\right) \\
    &=\frac{1}{2}\times (1/\text{z\_c\_logsigma.exp()}) \times \text{z\_c\_logsigma\_hat\_j[i].exp()}
\end{aligned}
$$
In the third term, 
$$
\begin{aligned}
-\frac{1}{2}(\pmb{\mu}_j^{\hat{\mathbf{z}}_c} - \pmb{\mu}^{\mathbf{z}_c})^\top{(\pmb{\Sigma}^{\mathbf{z}_c})}^{-1}(\pmb{\mu}_j^{\hat{\mathbf{z}}_c} - \pmb{\mu}^{\mathbf{z}_c}) &=  -\frac{1}{2} (\text{z\_c\_mu\_hat\_j} - \text{z\_c\_mu})^2 \times (1/\text{z\_c\_logsigma.exp()})
\end{aligned}
$$
We can now summarize our bm loss of this form in our PyTorch codes:
$$
\begin{aligned}
\text{self.loss\_bm} = - (1 / 2) & * ((\text{z\_c\_logsigma\_hat\_j} - \text{z\_c\_logsigma}) \\
                                    &- ((1 / \text{z\_c\_logsigma.exp()}) * \text{z\_c\_logsigma\_hat\_j.exp()}) \\ 
                                    &+ (\text{z\_c\_mu\_hat\_j} - \text{z\_c\_mu}) ** 2 * (1 / \text{z\_c\_logsigma.exp()})).\text{sum(1)}.\text{mean()}
\end{aligned}
$$



## Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@article{xia2022pluralistic,
  title={Pluralistic Image Completion with Probabilistic Mixture-of-Experts},
  author={Xia, Xiaobo and Yang, Wenhao and Ren, Jie and Li, Yewen and Zhan, Yibing and Han, Bo and Liu, Tongliang},
  journal={arXiv preprint arXiv:2205.09086},
  year={2022}
}
```



## Acknowledgments

*The authors would give special thanks to Mingrui Zhu (Xidian University), Zihan Ding (Princeton University), and Chenlai Qian (Southeast University) for helpful discussions and comments.* 



## Contact

This repo is currently maintained by Xiaobo Xia and Wenhao Yang, which is only for academic research use. If you have any problems about the implementation of our code, feel free to contact wenhaoyang.alpha@gmail.com. Discussions and questions about our paper are welcome via xiaoboxia.uni@gmail.com. 
