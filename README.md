审稿人1

Due to the character limit, the experimental results table is provided via an anonymous link, which is allowed by the ICML rules.

# Concerns about Prop. 3.3
Thank you so much for your insightful and valuable comments. Prop. 3.2 and 3.3 aim to explain why GDA-based methods outperform Cache-based methods under certain assumptions, providing deeper insights beyond the methods themselves. 

## Common Assumptions in Prop. 3.2 and 3.3
Both Prop. 3.2 and 3.3 rely on assumptions about the original zero-shot prediction probabilities. For Prop. 3.2, we assume the existence of high-confidence samples where the predicted labels match the true labels (see Equation 40 of [a]). Similarly, Prop. 3.3 assumes a sufficient number of high-confidence samples where pseudo-labels (from zero-shot predictions) align with the true labels. For these samples, GDA-based methods can better capture the underlying data distribution. This assumption will be explicitly stated in our paper.

## Low-Confidence Samples
For low-confidence samples, unreliable zero-shot predictions can be improved through techniques like collecting true labels via human feedback or using alternative refinement methods. These strategies help the GDA-based method better capture the true data distribution, making it more effective than the Cache-based approach.


## Setting $h_t$ to $\sqrt{n_t}$
While increasing the cache capacity can reduce the error in Proposition 2, this approach is impractical in resource-constrained environments. Setting $h_t$ to $\sqrt{n_t}$ would result in additional cache overhead that grows with the number of test samples.


# Experimental Analyses
We separated the ImageNetV2 to clearly analyze potential limitations. It will eventually be reintegrated into Table 2.

# Essential References
We have now added a discussion regarding paper [b]. Compared to [b], the proposed method differs in the following aspects:
- Different Task: [b] focuses on the few-shot scenario, whereas our method targets test-time adaptation.
- Different Distribution Estimation: As noted, our approach involves a series of modifications to handle test-time adaptation tasks. Specifically, we (1) replace the ground truth label with a pseudo label and (2) update the mean and covariance online. We think that both modifications are nontrivial.
- Adaptive Fusion of Zero-Shot and GDA Classifiers: Section 3.3 introduces an adaptive fusion of zero-shot and test-time classifiers to mitigate the unreliability risks of the GDA-based classifier. This is not covered in [b].
- Our main contribution is comparing our method with previous TTA approaches, not [b]. We propose a new paradigm for test-time adaptation, as an alternative to previous cache-based methods.


# Comparison to Naive Extensions of TDA
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


# More baselines. 
Since DMN[c] and DPE[d] used LLM-generated prompts, direct comparison is unfair. DMN[c] augmented CLIP prompts (Section 4.1 in [c]), and DPE[d] did the same (Table C8, page 20 in [d]). Both our work and TDA use the original CLIP prompt. Therefore, we compared our method with concurrent approaches using the same settings[a，e,f], our method still shows superior performance with an average improvement of about 1%.


| Method           | Aircraft | Caltech101 | Cars  | DTD   | EuroSAT | Flower102 | Food101 | Pets   | SUN397 | UCF101 | Average |
|:----------------:|:--------:|:----------:|:-----:|:-----:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| **CLIP-ViT-B/16** |  23.22   |   93.55    | 66.11 | 45.04 | 50.42   |  66.99   |  82.86  |  86.92 |  65.63 |  65.16 |  64.59  |
| **BoostAdapter[a]**  | 27.45 | 94.77 | 69.30 | 45.69 | 61.22 | 71.66 | 87.17 | 89.51 | 68.09 | 71.93 | 68.68 |
| **HisTPT[e]**        | 26.90 | 94.50 | 69.20 | 48.90 | 49.70 | 71.20 | 89.30 | 89.10 | 67.20 | 70.10 | 67.60 |
| **ZERO[f]**          | 25.21 | 93.66 | 68.04 | 46.12 | 34.33 | 67.68 | 86.53 | 87.75 | 65.03 | 67.77 | 64.21 |
| **Dota**         |  26.25   |   94.16    | 69.56 | 47.64 | 62.78   |  75.23   |  87.08  |  92.01 |  69.80 |  72.54 |  69.71  |


# Stability of Inversion Process of $\Sigma$
To enhance stability, we initialize $\Sigma$ with the identity matrix (Eq. 5) and apply shrinkage regularization to the precision matrix by adding an identity matrix (Page 4, right column, lines 210–216). These approach ensures well-conditioned eigenvalues. Furthermore, as you mentioned, sharing $\Sigma$ across all classes simplifies distribution estimation by reducing parameters.

# Alternative Choice of $\Sigma$ (Diagonal) 
Original experiments (Table 7) show that using a fixed diagonal $\Sigma$ without test-time updates results in a 2.39% performance drop.

[a] BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping. NeurIPS 2024.

[b] A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation. ICLR 2024.

[c] Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models. CVPR 2024.

[d] Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models. NeurIPS 2024.

[e] Historical Test-time Prompt Tuning for Vision Foundation Models. NeurIPS 2024.

[f] Frustratingly Easy Test-Time Adaptation of Vision-Language Models. NeurIPS 2024.

# 审稿人2

# Experiments

## whether the last 50% test samples are easier.

To eliminate potential biases from data flow ordering, we expanded the seed range from 1 to [2, 3, 4, 5]. The results consistently show that DOTA performs better on the latter 50% of samples across different seeds. On the ImageNet dataset, the average accuracy for vit improves from 64.62% to 65.20%, and for rn50, it increases from 55.31% to 55.45%. Similarly, on other datasets, the last 50% of test samples consistently outperform all test samples, with an average improvement about 1.1%.



| seed         |    2    |    3    |    4    |    5    |  average  |
|:------------:|:-------:|:-------:|:-------:|:-------:|:---------:|
| **vit**      |  64.72  |  64.56  |  64.54  |  64.66  |  64.62    |
| **last 50%** |  65.52  |  64.96  |  65.12  |  65.20  |  65.20    |
| **rn50**     |  54.99  |  55.57  |  55.27  |  55.39  |  55.31    |
| **last 50%** |  55.04  |  55.74  |  54.94  |  56.08  |  55.45    |


## visualization of the online generated classifiers of different test steps.

@gbaehgeae 添加实验和分析
In order to better understand the impact of online incremental updates to the model's parameters on its performance, we apply the DOTA model to the ImageNet dataset but limit the number of samples that can update the covariance matrix and mean. Specifically, we use the first *num* samples to update the model, while subsequent samples do not contribute to the model's updates. As shown in the table, having more samples available for updating the model tends to improve its accuracy.
| num    | 10000  | 20000  | 30000  | 40000  | 50000  |
|--------|--------|--------|--------|--------|--------|
| imagenet | 70.05  | 70.57  | 70.66  | 70.69  | 70.69  |

## Compatibility with CoOp and MaPLe

Thank you for your insightful question. The proposed method is indeed orthogonal to the prompt tuning approaches such as CoOp or MaPLe. It is also straightforward to combine our method with these techniques. Specifically, methods like CoOp focus on learning a new prompt using few-shot samples to obtain a new few-shot classifier. In our case, we can easily replace the zero-shot classifier mentioned in the paper with a few-shot classifier derived from CoOp or MaPLe without affecting the application of our proposed method. This flexibility allows our approach to seamlessly integrate with existing prompt tuning methods for further enhancement.

@gbaehgeae 添加实验和分析

# Essential References

Thank you for your suggestion. We will cite and discuss the related work in detail in the main text.

# overcome the problem of wrong zero-shot prediction

The challenge of incorrect zero-shot predictions is indeed a common issue faced during test-time adaptation. Previous methods have attempted to mitigate the negative impact of inaccurate zero-shot predictions by filtering samples based on entropy or confidence scores to retain only reliable zero-shot predictions.

In this paper, our proposed method addresses this problem from a fundamental perspective. During the process of estimating the data distribution to obtain classifiers, the estimation process can be intuitively understood as a reweighting mechanism. Specifically, the zero-shot predicted probabilities (which are generally well-calibrated [a]) are utilized as weights to adjust the contributions of samples to different classes. This approach effectively reduces the influence of potentially inaccurate zero-shot predictions, as samples with lower confidence contribute less to the final distribution estimation.

# Setting of $\mu$

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

# 审稿人3

# Clarifying experimental misunderstandings

1. The original statement, "Dota does not significantly surpass TDA," may cause misunderstanding. Our intention was to convey that, unlike the substantial performance gains demonstrated in Tables 1 and 2, Dota does not clearly outperform TDA on the ImageNetV2 dataset. We will revise the phrase to: "Dota performs slightly worse than TDA."

2. The modest improvement observed under ResNet-50 is supported by the experimental results presented in Tables 1 and 2. For example, in Table 1, the ViT-B/16 architecture shows an average improvement of approximately 2.2%, while the improvement under ResNet-50 is smaller at around 1.7%. Additionally, Table 2 further emphasizes this phenomenon, where the ViT-B/16 architecture achieves an average improvement of about 1.2%, compared to only approximately 0.3% for ResNet-50.

# Comparison with BoostAdapter

BoostAdapter, like TDA, is a Cache-based method with a similar fundamental framework. The difference lies in further streamlining TDA, such as removing the Negative Cache. Additionally, during testing, BoostAdapter constructs more reliable Cache samples through data augmentation. Here, we provide a performance comparison with BoostAdapter.  From the experimental results, it can be seen that the proposed method still outperforms BoostAdapter, with an average performance improvement of approximately 1%.


| Method           | Aircraft | Caltech101 | Cars  | DTD   | EuroSAT | Flower102 | Food101 | Pets   | SUN397 | UCF101 | Average |
|:----------------:|:--------:|:----------:|:-----:|:-----:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| **CLIP-ViT-B/16** |  23.22   |   93.55    | 66.11 | 45.04 | 50.42   |  66.99   |  82.86  |  86.92 |  65.63 |  65.16 |  64.59  |
| **BoostAdapter** |  27.45   |   94.77    | 69.30 | 45.69 | 61.22   |  71.66   |  87.17  |  89.51 |  68.09 |  71.93 |  68.68  |
| **Dota**         |  26.25   |   94.16    | 69.56 | 47.64 | 62.78   |  75.23   |  87.08  |  92.01 |  69.80 |  72.54 |  69.71  |




# The improvements are limited in some scenarios 

Designing a method applicable to all scenarios can be challenging. We have intentionally clarified the limitations of the proposed method in the paper to explicitly define its specific application scope. Even though, when considering the overall average performance of the model, the proposed method still demonstrates significant improvements over the compared methods. Therefore, although there may be certain limitations on specific datasets like ImageNet-V2 and certain network backbones like ResNet-50, the overall effectiveness of the method remains evident. Moreover, we believe that acknowledging these limitations provides a valuable foundation for further research and improvement.

# Theoretical contribution of DOTA

The primary purpose of the proposed analysis is to offer a fresh perspective for conducting test-time adaptation, aimed at highlighting the limitations of current cache-based test-time adaptation methods. By introducing this new viewpoint, we aim to better understand and address the inherent challenges associated with these existing approaches. This provides us with valuable insights, namely that we should not simply store the representations of test samples in the cache. Instead, we should make better use of these representations to handle potential errors when the cache capacity is limited.

# Base method of DOTA

The proposed DOTA has not been combined with any training-free test-time prompt learning methods. Our work and experiments are conducted based on the original base prompt of CLIP. 

@skylineeeeen 增加实验和分析，在这里我理解他的意思是，替换一下原始的prompt。可以把基于coop的那个实验拿过来。
To ensure a fair comparison with the DMN method, we also used the text prompts generated by the large model as input to the text encoder, leading to the following results. As shown, our DOTA method still outperforms DMN in terms of performance.

| Dataset   |  fgvc  | caltech101 | cars  | dtd   | eurosat | flower | food101 | pets  | sun397 | ucf101 | average |
|:---------:|:-----:|:----------:|:-----:|:-----:|:-------:|:------:|:-------:|:-----:|:------:|:------:|:-------:|
| **DMN**   |   30.03 |    95.38   | 67.96 | 55.85 |  59.43  | 74.49  |  85.08  | 92.04 | 70.18  | 72.51  |  70.30  |
| **DOTA**  |   29.82  |    94.85  | 69.06 | 55.97 |  58.35  | 77.06  |  87.07  | 92.40  | 70.97  | 74.86  |  71.04   |





# Does DOTA consider the negative caches?

DOTA uses a different approach compared to TDA. Unlike TDA, where caches need to be set, DOTA does not require setting up caches. Instead, DOTA leverages test samples to estimate the data distribution of the test samples themselves. This approach allows DOTA to operate without distinguishing between positive and negative caches, focusing solely on accurately estimating the test sample distribution.

The proposed method implicitly considers negative information by utilizing zero-shot probability estimation for data distribution. During the distribution estimation process, if a sample's zero-shot prediction probability for a certain category is low, its contribution to the corresponding data distribution estimation becomes minimal. This mechanism ensures that negative information is naturally accounted for in the estimation process.


# Some visualization results of DOTA.

@skylineeeeen 增加实验和分析
In order to better understand the impact of online incremental updates to the model's parameters on its performance, we apply the DOTA model to the ImageNet dataset but limit the number of samples that can update the covariance matrix and mean. Specifically, we use the first *num* samples to update the model, while subsequent samples do not contribute to the model's updates. As shown in the table, having more samples available for updating the model tends to improve its accuracy.

| num    | 10000  | 20000  | 30000  | 40000  | 50000  |
|--------|--------|--------|--------|--------|--------|
| imagenet | 70.05  | 70.57  | 70.66  | 70.69  | 70.69  |



# 审稿人4

Thanks for your positive comments.

# Solution for potential poisoning attack.

We appreciate your concern regarding the potential vulnerability of the DOTA algorithm to poisoning attacks during the continual learning process. To address your concern, we would like to respond with the following points:

Detection of Adversarial Samples Using Existing Methods: We can use existing methods to detect potential adversarial samples and remove these samples during the learning process to avoid their negative impact on adaptation during testing.

Improvements to Enhance Detection: We can to extend the DOTA framework to further improve robustness against adversarial attacks. One approach is to model the distribution of adversarial samples explicitly, allowing the system to distinguish between normal data and poisoned data more effectively.
