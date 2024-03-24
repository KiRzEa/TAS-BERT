import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from TAS_BERT_joint import convert_to_feature

model.to('cuda')

def extract_target(ner_predict):
  entities = []
  for i, sentence in enumerate(test):
    tokens = sentence.tokens
    predict = ner_predict[i]
    for j, token in enumerate(tokens):
      if j == 0:
        continue
      if predict[j] != 0:
        label = label_list[predict[j]]
        if label[0] == 'B':
          entities.append([token, label[2:]])
        elif label[0] == 'I':
          entities[-1][0] += ' ' + token
  return entities


def inference(text, compose_set, model):
  examples = processor.create_inference_example(text, compose_set)
  test = convert_to_feature(examples, 128, tokenizer, 'word_split')

  # Create a dataset from the examples
  all_input_ids = torch.tensor([f.input_ids for f in test], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in test], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in test], dtype=torch.long)
  all_ner_mask = torch.tensor([f.ner_mask for f in test], dtype=torch.long)

  test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ner_mask)
  # Create a data loader
  dataloader = DataLoader(test_data, batch_size=len(compose_set), shuffle=False)

  for batch in dataloader:
    all_input_ids, all_input_mask, all_segment_ids, all_ner_mask = batch
    all_input_ids = all_input_ids.to('cuda')
    all_input_mask = all_input_mask.to('cuda')
    all_segment_ids = all_segment_ids.to('cuda')
    all_ner_mask = all_ner_mask.to('cuda')
    logits, ner_predict = model(all_input_ids, all_segment_ids, all_input_mask, None, None, all_ner_mask)
  
  triplets = []
  outputs = np.argmax(torch.softmax(logits, dim=-1).detach().cpu().numpy(), axis=-1)
  indices = np.where(outputs==1)
  for idx in indices[0]:
    print(ner_predict[idx])
    target = extract_target(ner_predict[idx])
    aspect = '#'.join(compose_set[idx].upper().split()[0:2])
    sentiment = compose[idx].split()[-1]
    triplets.append((target, aspect, sentiment))
  return triplets
