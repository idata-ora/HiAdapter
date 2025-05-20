<h2 class="papername">HiAdapter: Histopathology-induced Adapter for Pathology Foundation Models </h2>


## Updates
+ TODO: Add code ✔

+ TODO: Add OSPatch

## Abstract

With the rapid development of pathology foundation models, there is a growing demand for efficient fine-tuning strategies tailored to downstream tasks, such as tissue classification. However, existing parameter-efficient fine-tuning approaches are largely task-agnostic and exhibit limited generalization to histopathological images, particularly for cancers not seen during pretraining, primarily due to substantial stain variability and the complexity of tissue microenvironments. To address these challenges, we present **Histopathology-induced Adapter (HiAdapter)**, which incorporates domain-specific insights into the staining and imaging mechanisms of histopathology. HiAdapter reconstructs stain-invariant representations and integrates morphology-aware features through dedicated adapter modules. Additionally, we introduce a Pathology Prototypical Contrastive Loss to mitigate inter-class similarity and intra-class heterogeneity. To assess the generalizability of HiAdapter to unseen cancers, we present **OSPatch**, the largest osteosarcoma dataset to date, with 56,178 annotated patches across six tissue categories. Experimental results on two public benchmarks and OSPatch, using three foundation models, demonstrate that HiAdapter consistently outperforms state-of-the-art baselines, achieving an average improvement of **2.43 in F1 score** and **2.09 in accuracy**.

The framework of the proposed HiAdapter:

<div align="center">
<img src='./assets/fig1.png' width='100%'>
</div>

## Enhanced Interpretability

Compared to other PEFT baselines, which generally display diffuse and inconsistent attention patterns with poor localization of biologically relevant structures, HiAdapter consistently generates sharper, spatially coherent maps that align with both localized cellular features and broader tissue architectures.

![Score](./assets/fig4.png)

## Data Link
+ SPIDER dataset: [here](https://github.com/HistAI/SPIDER)
+ OSPatch dataset: coming soon~

```python
├── OSPatch
│   ├── split
│   │   ├── train_fold_0.txt
│   │   └── val_fold_0.txt
│   │   └── test.txt
│   └── trainval
│       ├── 0
│       ├── 1
│       ├── ...
│       ├── nuclei
│       └── non
```


## Quick Start
```python
python train.py

