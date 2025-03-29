# reveiwer 1
## Comparison to Naive Extensions of TDA
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

| Method           | Aircraft | Caltech101 | Cars  | DTD   | EuroSAT | Flower102 | Food101 | Pets   | SUN397 | UCF101 | Average |
|:----------------:|:--------:|:----------:|:-----:|:-----:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| **CLIP-ViT-B/16** |  23.22   |   93.55    | 66.11 | 45.04 | 50.42   |  66.99   |  82.86  |  86.92 |  65.63 |  65.16 |  64.59  |
| **BoostAdapter[a]**  | 27.45 | 94.77 | 69.30 | 45.69 | 61.22 | 71.66 | 87.17 | 89.51 | 68.09 | 71.93 | 68.68 |
| **HisTPT[e]**        | 26.90 | 94.50 | 69.20 | 48.90 | 49.70 | 71.20 | 89.30 | 89.10 | 67.20 | 70.10 | 67.60 |
| **ZERO[f]**          | 25.21 | 93.66 | 68.04 | 46.12 | 34.33 | 67.68 | 86.53 | 87.75 | 65.03 | 67.77 | 64.21 |
| **Dota**         |  26.25   |   94.16    | 69.56 | 47.64 | 62.78   |  75.23   |  87.08  |  92.01 |  69.80 |  72.54 |  69.71  |

# reviewer 2
## whether the last 50% test samples are easier.

| seed         |    2    |    3    |    4    |    5    |  average  |
|:------------:|:-------:|:-------:|:-------:|:-------:|:---------:|
| **vit**      |  64.72  |  64.56  |  64.54  |  64.66  |  64.62    |
| **last 50%** |  65.52  |  64.96  |  65.12  |  65.20  |  65.20    |
| **rn50**     |  54.99  |  55.57  |  55.27  |  55.39  |  55.31    |
| **last 50%** |  55.04  |  55.74  |  54.94  |  56.08  |  55.45    |


## visualization of the online generated classifiers of different test steps.
| num    | 10000  | 20000  | 30000  | 40000  | 50000  |
|--------|--------|--------|--------|--------|--------|
| imagenet | 70.05  | 70.57  | 70.66  | 70.69  | 70.69  |

## Setting of $\mu$

| dataset   | imagenet | fgvc  | caltech101 | cars  | dtd   | flower | food101 | sun397 | ucf101 | average |
|:---------:|:--------:|:-----:|:----------:|:-----:|:-----:|:------:|:-------:|:------:|:------:|:-------:|
| **$\mu$ = 0** | 70.69    | 26.19 | 94.16      | 69.62 | 47.64 | 75.31  | 87.07   | 69.79  | 72.56  | 68.11   |
| **$\mu$ = 0.1** | 70.69  | 26.28 | 94.24      | 69.73 | 47.87 | 75.23  | 87.03   | 69.77  | 72.43  | 68.14   |
| **$\mu$ = 1**  | 70.71   | 26.55 | 94.20      | 69.68 | 47.87 | 75.03  | 87.03   | 69.80  | 72.38  | 68.14   |
| **randn(0,1)** | 64.7   | 24.93 | 90.43      | 66.14 | 47.34 | 71.54  | 86.73   | 64.71  | 68.78  | 65.03   |
| **clip_init**  | 70.68  | 25.65 | 94.32      | 69.47 | 47.87 | 74.58  | 87.02   | 69.69  | 72.09  | 67.93   |

# reviewer 3
## Comparison with BoostAdapter

| Method           | Aircraft | Caltech101 | Cars  | DTD   | EuroSAT | Flower102 | Food101 | Pets   | SUN397 | UCF101 | Average |
|:----------------:|:--------:|:----------:|:-----:|:-----:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| **CLIP-ViT-B/16** |  23.22   |   93.55    | 66.11 | 45.04 | 50.42   |  66.99   |  82.86  |  86.92 |  65.63 |  65.16 |  64.59  |
| **BoostAdapter** |  27.45   |   94.77    | 69.30 | 45.69 | 61.22   |  71.66   |  87.17  |  89.51 |  68.09 |  71.93 |  68.68  |
| **Dota**         |  26.25   |   94.16    | 69.56 | 47.64 | 62.78   |  75.23   |  87.08  |  92.01 |  69.80 |  72.54 |  69.71  |


## Base method of DOTA

| Dataset   |  fgvc  | caltech101 | cars  | dtd   | eurosat | flower | food101 | pets  | sun397 | ucf101 | average |
|:---------:|:-----:|:----------:|:-----:|:-----:|:-------:|:------:|:-------:|:-----:|:------:|:------:|:-------:|
| **DMN**   |   30.03 |    95.38   | 67.96 | 55.85 |  59.43  | 74.49  |  85.08  | 92.04 | 70.18  | 72.51  |  70.30  |
| **DOTA**  |   29.82  |    94.85  | 69.06 | 55.97 |  58.35  | 77.06  |  87.07  | 92.40  | 70.97  | 74.86  |  71.04   |


## Some visualization results of DOTA.

| num    | 10000  | 20000  | 30000  | 40000  | 50000  |
|--------|--------|--------|--------|--------|--------|
| imagenet | 70.05  | 70.57  | 70.66  | 70.69  | 70.69  |

