# Fact and scientific claim verification with Natural Language Understanding and reasoning
In this project, we propose to study and improve the reasoning ability of fact verification models over structured and unstructured information by finding precise evidence to support or dismiss a claim. Our goal is to provide researchers a framework to generate datasets, evaluate models, and identify adversarial vulnerabilities of their fact verification models.

Datasets - CLIMATE-FEVER
Claim generation using a Google T5 model finetuned on the SQuAD 1.1 dataset, pretrained Sense2Vec
Claim augmentation using negative claim generation model fine-tuned with WikiFactCheck-English from HuggingFace

Goal:
1.  Create a large publicly available dataset for verification of climate change-related claims by generating and augmenting claims in CLIMATE_FEVER
2.  Analyse the performance of an ensemble of fact verification models on the baseline and augmented datasets
3.  Analyse the fact verification model's robustness to adversarial queries and identify vulnerablities of the model 


## Claim generation and data augmentation with negative claim generation
While the widely popular dataset for fact verification, FEVER has 185,445 claims manually verified against the introductory sections of Wikipedia pages and classified as SUPPORTED, REFUTED or NOTENOUGHINFO, CLIMATE-FEVER only contains 1,535 real-world claims regarding climate-change. Collecting hand-annotated claims, especially related to climate change is extremely challenging. We want to test the performace of fact verification models on the original CLIMATE-FEVER dataset and compare it with models trained with the generated claims. The dataset features challenging claims that relate multiple facets and disputed cases of claims where both supporting and refuting evidence are present, making claim generation on this data challenging.

Progress Report:
1. Claim generation and data augmentation on CLIMATE-FEVER
2. One baseline model trained on CLIMATE-FEVER - there are no baselines for this dataset
3. Analysis of adversarial robustness of one of the three popular fact verification model
