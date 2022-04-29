import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

f = open("/content/climate_dev_processed.json")
climate_dev = json.load(f)

for claim in climate_dev:
  if claim['label']=='REFUTED':
    claim['label'] = 0
  if claim['label']=='SUPPORTED':
    claim['label'] = 1
  if claim['label']=='NEI':
    claim['label'] = 2

claim_list = []
evidence_list = []
label_list = []
for claim in climate_dev:
  claim_list.append(claim['claim'])
  evidence_list.append(claim['context'])
  label_list.append(claim['label'])

test_dict = {'claim':claim_list, 'evidence':evidence_list, 'label':label_list}

test_dataset = Dataset.from_dict(test_dict)

tokenizer = AutoTokenizer.from_pretrained("amandakonet/climatebert-fact-checking")

def tokenize_function(examples):
  return tokenizer(examples["claim"],examples["evidence"],padding="max_length", truncation=True)

test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)

test_tokenized_datasets = test_tokenized_datasets.remove_columns(["claim"])
test_tokenized_datasets = test_tokenized_datasets.remove_columns(["evidence"])
test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
test_tokenized_datasets.set_format("torch")

eval_dataloader = DataLoader(test_tokenized_datasets, batch_size=16)

model = AutoModelForSequenceClassification.from_pretrained("amandakonet/climatebert-fact-checking")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

model.eval()
eval_labels = []
with torch.no_grad():
   for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch).logits
   
        label_mapping = ['SUPPORTED', 'REFUTED', 'NEI']
        labels = [label_mapping[score_max] for score_max in outputs.argmax(dim=1)]
        eval_labels += labels
        print(labels)

labels = []
for claim in climate_dev:
  labels.append(claim['label'])

pred_labels = []
for label in eval_labels:
  if label=='NEI':
    pred_labels.append(2)
  if label=='SUPPORTED':
    pred_labels.append(0)
  if label=='REFUTED':
    pred_labels.append(1)

correct_idx = []
correct = 0
for i in range(len(labels)):
  if pred_labels[i]==labels[i]:
    correct_idx.append(i)
    correct += 1
print(correct/len(labels))