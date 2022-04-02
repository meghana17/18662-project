## Fact and scientific claim verification with Natural Language Understanding and reasoning
In this project, we propose to study and improve the reasoning ability of fact verification models over structured and unstructured information by finding precise evidence to support or dismiss a claim. Our goal is to provide researchers a framework to generate datasets, evaluate models, and identify adversarial vulnerabilities of their fact verification models.

Datasets - CLIMATE-FEVER
Claim generation using a Google T5 model finetuned on the SQuAD 1.1 dataset, pretrained Sense2Vec
Claim augmentation using negative claim generation model fine-tuned with WikiFactCheck-English from HuggingFace

Goal:
1.  Create a large publicly available dataset for verification of climate change-related claims by generating and augmenting claims in CLIMATE_FEVER
2.  Analyse the performance of an ensemble of fact verification models on the baseline and augmented datasets
3.  Analyse the fact verification model's robustness to adversarial queries and identify vulnerablities of the model 
