import json
import random

f = open("/content/climate_fever.json")
claim_json = json.load(f)

climate_processed = []
i = 3135
for claim_dict in claim_json:
  claim = claim_dict['claim']
  evidences = claim_dict['evidences']
  for evidence in evidences:
    id = i
    label = evidence['evidence_label']
    if label=='SUPPORTS':
      label = 'SUPPORTED'
    if label=='REFUTES':
      label = 'REFUTED'
    if label=='NOT_ENOUGH_INFO':
      label='NEI'
    context = evidence['evidence']
    constructed_claim = {"id":id, "context":context,"claim":claim, "label":label }
    climate_processed.append(constructed_claim)
    i += 1

# Split the dataset into train-test
random.shuffle(climate_processed)
climate_train_processed = climate_processed[:6100]
climate_dev_processed = climate_processed[6100:]

with open('climate_train_processed.json', 'w', encoding='utf-8') as f:
    json.dump(climate_train_processed, f, ensure_ascii=False, indent=4)

with open('climate_dev_processed.json', 'w', encoding='utf-8') as f:
    json.dump(climate_dev_processed, f, ensure_ascii=False, indent=4)
