import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

f = open("/content/climate_train_processed.json")
climate_train = json.load(f)

f = open("/content/climate_dev_processed.json")
climate_dev = json.load(f)

f = open("/content/drive/MyDrive/18662/Project/SUPPORTED_claims.json")
supported = json.load(f)

f = open("/content/drive/MyDrive/18662/Project/REFUTED_claims.json")
refuted = json.load(f)

for claim in refuted:
  del claim['replace_type']

generated_claims = climate_train + climate_dev + supported + refuted
random.shuffle(generated_claims)

for claim in generated_claims:
  if claim['label']=='REFUTED':
    claim['label'] = 0
  if claim['label']=='SUPPORTED':
    claim['label'] = 1
  if claim['label']=='NEI':
    claim['label'] = 2

train_split = int(0.8*len(generated_claims))
train_claims = generated_claims[:train_split]
test_claims = generated_claims[train_split:]

claim_list = []
evidence_list = []
label_list = []
for claim in train_claims:
  claim_list.append(claim['claim'])
  evidence_list.append(claim['context'])
  label_list.append(claim['label'])
train_dict = {'claim':claim_list, 'evidence':evidence_list, 'label':label_list}

claim_list = []
evidence_list = []
label_list = []
for claim in test_claims:
  claim_list.append(claim['claim'])
  evidence_list.append(claim['context'])
  label_list.append(claim['label'])
test_dict = {'claim':claim_list, 'evidence':evidence_list, 'label':label_list}

train_dataset = Dataset.from_dict(train_dict)
test_dataset = Dataset.from_dict(test_dict)

tokenizer = AutoTokenizer.from_pretrained("amandakonet/climatebert-fact-checking")

def tokenize_function(examples):
  return tokenizer(examples["claim"],examples["evidence"],padding="max_length", truncation=True)

train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)

train_tokenized_datasets = train_tokenized_datasets.remove_columns(["claim"])
train_tokenized_datasets = train_tokenized_datasets.remove_columns(["evidence"])
test_tokenized_datasets = test_tokenized_datasets.remove_columns(["claim"])
test_tokenized_datasets = test_tokenized_datasets.remove_columns(["evidence"])
train_tokenized_datasets = train_tokenized_datasets.rename_column("label", "labels")
test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
train_tokenized_datasets.set_format("torch")
test_tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(train_tokenized_datasets, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(test_tokenized_datasets, batch_size=8)

# Load the model to fine-tune here
# <Options>
# BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
# RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
# AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=3)
# DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained("amandakonet/climatebert-fact-checking")

num_epochs = 10
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# Train (fine-tune) the model
progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
