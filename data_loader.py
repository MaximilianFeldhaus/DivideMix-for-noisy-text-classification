from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
import numpy as np
import json
import os
import torch
from torchnet.meter import AUCMeter


def collate_batch(batch):
    if isinstance(batch[0], tuple):
        # For labeled data
        inputs, targets, *rest = zip(*batch)
        max_len = max(inp['input_ids'].size(0) for inp in inputs)

        padded_inputs = {
            'input_ids': pad_sequence([inp['input_ids'] for inp in inputs], batch_first=True, padding_value=0, total_length=max_len),
            'attention_mask': pad_sequence([inp['attention_mask'] for inp in inputs], batch_first=True,
                                           padding_value=0, total_length=max_len),
        }

        if 'token_type_ids' in inputs[0]:
            padded_inputs['token_type_ids'] = pad_sequence([inp['token_type_ids'] for inp in inputs], batch_first=True,
                                                           padding_value=0)

        targets = torch.tensor(targets)

        if rest:
            return (padded_inputs, targets) + tuple(torch.tensor(r) for r in rest)
        return padded_inputs, targets
    else:
        # For unlabeled data
        max_len = max(inp['input_ids'].size(0) for inp in batch)

        padded_inputs = {
            'input_ids': pad_sequence([inp['input_ids'] for inp in batch], batch_first=True, padding_value=0,
                                      total_length=max_len),
            'attention_mask': pad_sequence([inp['attention_mask'] for inp in batch], batch_first=True, padding_value=0,
                                           total_length=max_len),
        }

        if 'token_type_ids' in batch[0]:
            padded_inputs['token_type_ids'] = pad_sequence([inp['token_type_ids'] for inp in batch], batch_first=True,
                                                           padding_value=0)

        return padded_inputs

class text_dataset(Dataset):
    def __init__(self, r, noise_mode, root_dir, tokenizer, mode, noise_file='', pred=[], probability=[],
                 log=''):
        self.tokenizer = tokenizer # initiliaze tokenizer
        self.r = r # noise ratio
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = 512
        self.transition = {}

        if self.mode == 'test':
            self.test_data, self.test_label = self.load_data(os.path.join(root_dir, 'test.json'))
        else:
            self.train_data, self.train_label = self.load_data(os.path.join(root_dir, 'train.json'))

            if os.path.exists(noise_file):
                self.noise_label = json.load(open(noise_file, "r"))
            else:
                self.noise_label = self.inject_noise(self.train_label, noise_mode)
                print("save noisy labels to %s ..." % noise_file)
                json.dump(self.noise_label, open(noise_file, "w"))

            if self.mode == 'all':
                self.train_data = self.train_data
                self.noise_label = self.noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(self.noise_label) == np.array(self.train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    log.write('Number of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = [self.train_data[i] for i in pred_idx]
                self.noise_label = [self.noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        return texts, labels

    def inject_noise(self, labels, noise_mode):
        noisy_labels = labels.copy()
        num_samples = len(labels)
        num_classes = max(labels) + 1
        num_noise = int(self.r * num_samples)
        noise_idx = random.sample(range(num_samples), num_noise)

        for idx in noise_idx:
            if noise_mode == 'sym':
                noisy_labels[idx] = random.randint(0, num_classes - 1)
            elif noise_mode == 'asym':
                noisy_labels[idx] = self.transition.get(labels[idx], labels[idx])

        return noisy_labels

    #changed in order to handle text data
    def __getitem__(self, index):
        if self.mode == 'labeled':
            text, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            inputs = self.tokenizer(text, truncation=True, return_tensors='pt') # truncate=true for dynamic batching process
            inputs = {k: v.squeeze(0) for k, v in inputs.items()} # batch dimension removal
            return inputs, target, prob
        elif self.mode == 'unlabeled':
            text = self.train_data[index]
            inputs = self.tokenizer(text, truncation=True, return_tensors='pt') # truncate=true for dynamic batching process
            inputs = {k: v.squeeze(0) for k, v in inputs.items()} # batch dimension removal
            return inputs
        elif self.mode == 'all':
            text, target = self.train_data[index], self.noise_label[index]
            inputs = self.tokenizer(text, truncation=True, return_tensors='pt') # truncate=true for dynamic batching process
            inputs = {k: v.squeeze(0) for k, v in inputs.items()} # batch dimension removal
            return inputs, target, index
        elif self.mode == 'test':
            text, target = self.test_data[index], self.test_label[index]
            inputs = self.tokenizer(text, truncation=True, return_tensors='pt') # truncate=true for dynamic batching process
            inputs = {k: v.squeeze(0) for k, v in inputs.items()} # batch dimension removal
            return inputs, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class text_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = text_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                       root_dir=self.root_dir, tokenizer=self.tokenizer, mode="all",
                                       noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_batch)
            return trainloader

        elif mode == 'train':
            labeled_dataset = text_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                           root_dir=self.root_dir, tokenizer=self.tokenizer, mode="labeled",
                                           noise_file=self.noise_file, pred=pred, probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_batch)

            unlabeled_dataset = text_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                             root_dir=self.root_dir, tokenizer=self.tokenizer, mode="unlabeled",
                                             noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_batch)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = text_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, tokenizer=self.tokenizer, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_batch)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = text_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, tokenizer=self.tokenizer, mode='all',
                                        noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_batch)
            return eval_loader