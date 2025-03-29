Due to the character limit, the experimental results table is provided via an anonymous link, which is allowed by the ICML rules.
# reveiwer 1
## Comparison to Naive Extensions of TDA
We conducted experiments on the ImageNet dataset by increasing cache size for TDA and setting a fixed entropy threshold. Results show these changes do not significantly improve performance, consistent with the ablation study in [a] and TDA.

We tested different static cache sizes from the start, but accuracy remained similar to the original 69.50%. We tested dynamic cache size adjustments by gradually increasing cache sizes during the training process. However, the accuracy remained similar, around 69.35%. We also experimented with various entropy threshold settings, but observed no notable accuracy improvement over 69.50%. 


| pos \ neg | 2    | 3    | 4    |
|:--------:|:----:|:----:|:----:|
| **3**        | 69.50 | 69.28 | 69.15 |
| **4**        | 69.49 | 69.53 | 69.28 |
| **5**        | 69.46 | 69.53 | 69.35 |

| pos \ neg | 6     | 8     | 10    |
|:---------:|:-----:|:-----:|:-----:|
| **6**     | 68.98 | 68.57 | 68.24 |
| **8**     | 68.92 | 68.56 | 68.24 |
| **10**    | 68.80 | 68.70 | 68.20 |

| lower \ upper | 0.5  | 0.6  | 0.7  |
|:------------:|:----:|:----:|:----:|
| **0.1**          | 69.72 | 69.57 | 69.54 |
| **0.2**          | 69.47 | 69.55 | 69.53 |
| **0.3**          | 69.25 | 69.36 | 69.34 |


## More baselines. 
Since DMN[c] and DPE[d] used LLM-generated prompts, direct comparison is unfair. DMN[c] augmented CLIP prompts (Section 4.1 in [c]), and DPE[d] did the same (Table C8, page 20 in [d]). Both our work and TDA use the original CLIP prompt. Therefore, we compared our method with concurrent approaches using the same settings[a，e,f], our method still shows superior performance with an average improvement of about 1%.


| Method           | Aircraft | Caltech101 | Cars  | DTD   | EuroSAT | Flower102 | Food101 | Pets   | SUN397 | UCF101 | Average |
|:----------------:|:--------:|:----------:|:-----:|:-----:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| **CLIP-ViT-B/16** |  23.22   |   93.55    | 66.11 | 45.04 | 50.42   |  66.99   |  82.86  |  86.92 |  65.63 |  65.16 |  64.59  |
| **BoostAdapter[a]**  | 27.45 | 94.77 | 69.30 | 45.69 | 61.22 | 71.66 | 87.17 | 89.51 | 68.09 | 71.93 | 68.68 |
| **HisTPT[e]**        | 26.90 | 94.50 | 69.20 | 48.90 | 49.70 | 71.20 | 89.30 | 89.10 | 67.20 | 70.10 | 67.60 |
| **ZERO[f]**          | 25.21 | 93.66 | 68.04 | 46.12 | 34.33 | 67.68 | 86.53 | 87.75 | 65.03 | 67.77 | 64.21 |
| **Dota**         |  26.25   |   94.16    | 69.56 | 47.64 | 62.78   |  75.23   |  87.08  |  92.01 |  69.80 |  72.54 |  69.71  |


[a] BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping. NeurIPS 2024.

[b] A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation. ICLR 2024.

[c] Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models. CVPR 2024.

[d] Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models. NeurIPS 2024.

[e] Historical Test-time Prompt Tuning for Vision Foundation Models. NeurIPS 2024.

[f] Frustratingly Easy Test-Time Adaptation of Vision-Language Models. NeurIPS 2024.

# reviewer 2
## whether the last 50% test samples are easier.

To eliminate potential biases from data flow ordering, we expanded the seed range from 1 to [2, 3, 4, 5]. The results consistently show that DOTA performs better on the latter 50% of samples across different seeds. On the ImageNet dataset, the average accuracy for vit improves from 64.62% to 65.20%, and for rn50, it increases from 55.31% to 55.45%. Similarly, on other datasets, the last 50% of test samples consistently outperform all test samples, with an average improvement about 1.1%.



| seed         |    2    |    3    |    4    |    5    |  average  |
|:------------:|:-------:|:-------:|:-------:|:-------:|:---------:|
| **vit**      |  64.72  |  64.56  |  64.54  |  64.66  |  64.62    |
| **last 50%** |  65.52  |  64.96  |  65.12  |  65.20  |  65.20    |
| **rn50**     |  54.99  |  55.57  |  55.27  |  55.39  |  55.31    |
| **last 50%** |  55.04  |  55.74  |  54.94  |  56.08  |  55.45    |


## visualization of the online generated classifiers of different test steps.

In order to better understand the impact of online incremental updates to the model's parameters on its performance, we apply the DOTA model to the ImageNet dataset but limit the number of samples that can update the covariance matrix and mean. Specifically, we use the first *num* samples to update the model, while subsequent samples do not contribute to the model's updates. As shown in the table, having more samples available for updating the model tends to improve its accuracy.
| num    | 10000  | 20000  | 30000  | 40000  | 50000  |
|--------|--------|--------|--------|--------|--------|
| imagenet | 70.05  | 70.57  | 70.66  | 70.69  | 70.69  |

## Compatibility with CoOp and MaPLe

Thank you for your insightful question. The proposed method is indeed orthogonal to the prompt tuning approaches such as CoOp or MaPLe. It is also straightforward to combine our method with these techniques. Specifically, methods like CoOp focus on learning a new prompt using few-shot samples to obtain a new few-shot classifier. In our case, we can easily replace the zero-shot classifier mentioned in the paper with a few-shot classifier derived from CoOp or MaPLe without affecting the application of our proposed method. This flexibility allows our approach to seamlessly integrate with existing prompt tuning methods for further enhancement.

## Setting of $\mu$

We chose the simplest approach to set $\mu$ to avoid introducing complex operations into the method. When the $\mu$ for all classes are initialized to 1, the classifier’s predictions form a uniform distribution, meaning all classes have equal probability. This corresponds to a state where the classifier has not yet begun learning and is essentially ignorant. We have also added additional ablation studies with various initialization methods, and the results indicate that the proposed method is generally robust to different initializations of $\mu$.  

Here, we conducted more tests on the selection of the $\mu$, using several datasets. The results are shown below, and it can be seen that the initialization of $\mu$ has almost no impact on the experimental results.

| dataset   | imagenet | fgvc  | caltech101 | cars  | dtd   | flower | food101 | sun397 | ucf101 | average |
|:---------:|:--------:|:-----:|:----------:|:-----:|:-----:|:------:|:-------:|:------:|:------:|:-------:|
| **$\mu$ = 0** | 70.69    | 26.19 | 94.16      | 69.62 | 47.64 | 75.31  | 87.07   | 69.79  | 72.56  | 68.11   |
| **$\mu$ = 0.1** | 70.69  | 26.28 | 94.24      | 69.73 | 47.87 | 75.23  | 87.03   | 69.77  | 72.43  | 68.14   |
| **$\mu$ = 1**  | 70.71   | 26.55 | 94.20      | 69.68 | 47.87 | 75.03  | 87.03   | 69.80  | 72.38  | 68.14   |
| **randn(0,1)** | 64.7   | 24.93 | 90.43      | 66.14 | 47.34 | 71.54  | 86.73   | 64.71  | 68.78  | 65.03   |
| **clip_init**  | 70.68  | 25.65 | 94.32      | 69.47 | 47.87 | 74.58  | 87.02   | 69.69  | 72.09  | 67.93   |

For the record, the *randn(0,1)* initialization refers to randomly initializing the parameters using a Gaussian distribution with a mean of 0 and a variance of 1, while the *clip_init* refers to using the vector representations obtained from the CLIP text encoder for each class as the initialization.

[a] Revisiting the calibration of modern neural networks.

# reviewer 3
## Comparison with BoostAdapter

BoostAdapter, like TDA, is a Cache-based method with a similar fundamental framework. The difference lies in further streamlining TDA, such as removing the Negative Cache. Additionally, during testing, BoostAdapter constructs more reliable Cache samples through data augmentation. Here, we provide a performance comparison with BoostAdapter.  From the experimental results, it can be seen that the proposed method still outperforms BoostAdapter, with an average performance improvement of approximately 1%.


| Method           | Aircraft | Caltech101 | Cars  | DTD   | EuroSAT | Flower102 | Food101 | Pets   | SUN397 | UCF101 | Average |
|:----------------:|:--------:|:----------:|:-----:|:-----:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| **CLIP-ViT-B/16** |  23.22   |   93.55    | 66.11 | 45.04 | 50.42   |  66.99   |  82.86  |  86.92 |  65.63 |  65.16 |  64.59  |
| **BoostAdapter** |  27.45   |   94.77    | 69.30 | 45.69 | 61.22   |  71.66   |  87.17  |  89.51 |  68.09 |  71.93 |  68.68  |
| **Dota**         |  26.25   |   94.16    | 69.56 | 47.64 | 62.78   |  75.23   |  87.08  |  92.01 |  69.80 |  72.54 |  69.71  |


## Base method of DOTA

The proposed DOTA has not been combined with any training-free test-time prompt learning methods. Our work and experiments are conducted based on the original base prompt of CLIP. 

To ensure a fair comparison with the DMN method, we also used the text prompts generated by the large model as input to the text encoder, leading to the following results. As shown, our DOTA method still outperforms DMN in terms of performance.

| Dataset   |  fgvc  | caltech101 | cars  | dtd   | eurosat | flower | food101 | pets  | sun397 | ucf101 | average |
|:---------:|:-----:|:----------:|:-----:|:-----:|:-------:|:------:|:-------:|:-----:|:------:|:------:|:-------:|
| **DMN**   |   30.03 |    95.38   | 67.96 | 55.85 |  59.43  | 74.49  |  85.08  | 92.04 | 70.18  | 72.51  |  70.30  |
| **DOTA**  |   29.82  |    94.85  | 69.06 | 55.97 |  58.35  | 77.06  |  87.07  | 92.40  | 70.97  | 74.86  |  71.04   |


## Some visualization results of DOTA.

In order to better understand the impact of online incremental updates to the model's parameters on its performance, we apply the DOTA model to the ImageNet dataset but limit the number of samples that can update the covariance matrix and mean. Specifically, we use the first *num* samples to update the model, while subsequent samples do not contribute to the model's updates. As shown in the table, having more samples available for updating the model tends to improve its accuracy.

| num    | 10000  | 20000  | 30000  | 40000  | 50000  |
|--------|--------|--------|--------|--------|--------|
| imagenet | 70.05  | 70.57  | 70.66  | 70.69  | 70.69  |

