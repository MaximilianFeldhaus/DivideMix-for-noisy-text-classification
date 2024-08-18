from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from transformers import BertTokenizer, BertForSequenceClassification
import nlpaug.augmenter.word as naw

parser = argparse.ArgumentParser(description='PyTorch Text Classification with DivideMix')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=2e-5, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--data_path', default='./data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='sst2', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Load tokenizer - necessary because of the changde feature extractor
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text augmentation
aug = naw.SynonymAug(aug_src='wordnet') # creates an augmenter that replaces words with their synonyms using WordNet

def augment_text(input_ids):
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    augmented_texts = [aug.augment(text) for text in texts] # usage of augmenter

    augmented_inputs = tokenizer(augmented_texts, padding=True, truncation=True, return_tensors="pt") # covnert back
    # into a format that BERT can handle

    return augmented_inputs['input_ids'].cuda()

#Change of the model architecture to a transformer (BERT)
def create_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_class)
    model = model.cuda()
    return model

#Perform MixUp on embedding space
def text_mixup(embeddings, targets, l):
    batch_size = embeddings.size(0)

    idx = torch.randperm(batch_size)
    mixed_embeddings = l * embeddings + (1 - l) * embeddings[idx]
    mixed_targets = l * targets + (1 - l) * targets[idx]

    return mixed_embeddings, mixed_targets

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, _ = unlabeled_train_iter.next()
        batch_size = inputs_x['input_ids'].size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x = {k: v.cuda() for k, v in inputs_x.items()}
        inputs_u = {k: v.cuda() for k, v in inputs_u.items()}
        labels_x, w_x = labels_x.cuda(), w_x.cuda()

        # Apply text augmentation
        aug_inputs_x = augment_text(inputs_x['input_ids'])
        aug_inputs_u = augment_text(inputs_u['input_ids'])

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u1 = net(input_ids=aug_inputs_u, attention_mask=inputs_u['attention_mask']).logits
            outputs_u2 = net2(input_ids=aug_inputs_u, attention_mask=inputs_u['attention_mask']).logits

            pu = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            ptu = pu ** (1 / args.T)

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(input_ids=aug_inputs_x, attention_mask=inputs_x['attention_mask']).logits

            px = torch.softmax(outputs_x, dim=1)
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T) # temperature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        # Get embeddings
        embeddings_x = net.bert(input_ids=aug_inputs_x,
                                attention_mask=inputs_x['attention_mask']).last_hidden_state[:, 0, :]
        embeddings_u = net.bert(input_ids=aug_inputs_u,
                                attention_mask=inputs_u['attention_mask']).last_hidden_state[:, 0, :]

        all_embeddings = torch.cat([embeddings_x, embeddings_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        mixed_embeddings, mixed_targets = text_mixup(all_embeddings, all_targets, args.alpha, l)

        logits = net.classifier(mixed_embeddings)
        logits_x = logits[:batch_size]
        logits_u = logits[batch_size:]

        Lx, Lu, lamb = criterion(logits_x, mixed_targets[:batch_size], logits_u, mixed_targets[batch_size:],
                                 epoch + batch_idx / num_iter, warm_up)

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item()))
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        loss = CEloss(outputs, labels)
        if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode == 'sym':
            L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            targets = targets.cuda()
            outputs1 = net1(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
            outputs2 = net2(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            targets = targets.cuda()
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
            loss = CE(outputs, targets)
            for b in range(inputs['input_ids'].size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

if args.dataset=='textdata':
    warm_up = 10
elif args.dataset=='textdata':
    warm_up = 30

loader = dataloader.text_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.AdamW(net1.parameters(), lr=args.lr)
optimizer2 = optim.AdamW(net2.parameters(), lr=args.lr)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()

all_loss = [[], []]  # save the history of losses from two networks

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 5:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)

    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])

        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

    test(epoch, net1, net2)
