import requests
import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
res = requests.get(f'{url}train-v2.0.json')

for file in ['train-v2.0.json', 'dev-v2.0.json']:
    res = requests.get(f'{url}{file}')
    # write to file
    with open(f'../QA/{file}', 'wb') as f:
        for chunk in res.iter_content(chunk_size=4):
            f.write(chunk)

def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa['answers']:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_squad('train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('dev-v2.0.json')

def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # Sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
    
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
optim = AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(3):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model.eval()
total_eval_loss = 0
for batch in tqdm(val_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
    loss = outputs.loss
    total_eval_loss += loss.item()

model_path = 'models/distilbert-custom'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# switch model out of training mode
model.eval()

#val_sampler = SequentialSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=16)

acc = []

# initialize loop for progress bar
loop = tqdm(val_loader)
# loop through batches
for batch in loop:
    # we don't need to calculate gradients as we're not training
    with torch.no_grad():
        # pull batched items from loader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)
        # make predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        # pull preds out
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        # calculate accuracy for both and append to accuracy list
        acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
# calculate average accuracy in total
acc = sum(acc)/len(acc)


print("T/F\tstart\tend\n")
for i in range(len(start_true)):
    print(f"true\t{start_true[i]}\t{end_true[i]}\n"
          f"pred\t{start_pred[i]}\t{end_pred[i]}\n")
