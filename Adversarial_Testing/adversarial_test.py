from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model

model = AutoModelForSequenceClassification.from_pretrained("amandakonet/climatebert-fact-checking")
tokenizer = AutoTokenizer.from_pretrained("amandakonet/climatebert-fact-checking")

features = tokenizer(['Scientists no estimated that global warming had increased the probability of local record-breaking monthly temperatures by a factor of 5 in 2002.'], 
                   ['(2002) Scientists estimated that global warming had increased the probability of local record-breaking monthly temperatures worldwide by a factor of 5'],  
                   padding='max_length', truncation=True, return_tensors="pt", max_length=512)

model.eval()
with torch.no_grad():
   scores = model(**features).logits
   label_mapping = ['REFUTED', 'SUPPORTED', 'NEI']
   labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
   print(labels)
