# Assertion Classification
This repository contains the source code and data files for [Trustworthy assertion classification through prompting](https://www.sciencedirect.com/science/article/pii/S1532046422001538).

## Abstract
Accurate identification of the presence, absence or possibility of relevant entities in clinical notes is important for healthcare professionals to quickly understand crucial clinical information. This introduces the task of assertion classification - to correctly identify the assertion status of an entity in the unstructured clinical notes. Recent rule-based and machine-learning approaches suffer from labor-intensive pattern engineering and severe class bias toward majority classes. To solve this problem, in this study, we propose a prompt-based learning approach, which treats the assertion classification task as a masked language auto-completion problem. We evaluated the model on six datasets. Our prompt-based method achieved a micro-averaged F-1 of 0.954 on the i2b2 2010 assertion dataset, with $\sim$1.8% improvements over previous works. In particular, our model showed excellence in detecting classes with few instances (few-shot). Evaluations on five external datasets showcase the outstanding generalizability of the prompt-based method to unseen data. To examine the rationality of our model, we further introduced two rationale faithfulness metrics: comprehensiveness and sufficiency. The results reveal that compared to the “pre-train, fine-tune” procedure, our prompt-based model has a stronger capability of identifying the comprehensive ($\sim$63.93%) and sufficient ($\sim$11.75%) linguistic features from free text. We further evaluated the model-agnostic explanations using LIME. The results imply a better rationale agreement between our model and human beings (71.93% in average F-1), which demonstrates the superior trustworthiness of our model.

## Usage
Datasets can be found under `/data`, source codes can be found under `/src`.

Dataset Statistics:

| Dataset | Total |
| :---    | ---:  |
| i2b2 2010 Train | 7,073 |
| i2b2 2010 Test  | 12,563 |
| i2b2 2012 Test  | 4,309 |
| BioScope | 7,605 |
| MIMIC-III | 5,000 |
| NegEx | 2,376 |
| Chia | 2,114 |

## Citation
If you find our work useful, please cite the following paper:
```
@article{wang2022trustworthy,
  title={Trustworthy assertion classification through prompting},
  author={Wang, Song and Tang, Liyan and Majety, Akash and Rousseau, Justin F and Shih, George and Ding, Ying and Peng, Yifan},
  journal={Journal of Biomedical Informatics},
  pages={104139},
  year={2022},
  publisher={Elsevier}
}
```

## Acknowledgement
This work is supported by the National Library of Medicine, USA under Award No. 4R00LM013001, Amazon Diagnostic Development Initiative 2022, Cornell Multi-investigator Seed Grant 2022, and the NSF AI Institute for Foundations of Machine Learning (IFML).

