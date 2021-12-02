# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from model import STCKAtten
from utils import dataset, metrics, config
import copy
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore") 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
device = torch.device(device)

def train(classifier, train_loader, dev_loader, epoch, lr, optimizer, criterion, network="cnn"):
    classifier.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        short_text_input, input_concepts, targets = batch.text[0], batch.concept[0], batch.label # batch.text and batch.concept must be indiced because of include_lengths being true in data.field 
        optimizer.zero_grad()

        short_text_input, input_concepts, targets = short_text_input.to(device), input_concepts.to(device), targets.to(device)

        output = classifier(short_text_input, input_concepts, network)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        if(not batch_idx % 10):
            logger.info('Epoch:%d, Idx:%d, Training Loss:%.4f', epoch, batch_idx, loss.item())

        train_loss += loss.item()

    dev_loss, accuracy, precision, recall, f1_score = test(classifier, dev_loader, criterion, network)
    train_loss /= len(train_loader)

    return train_loss, dev_loss, accuracy, precision, recall, f1_score

def test(classifier, loader, criterion, network="cnn"):
    classifier.eval()
    test_loss = 0
    target_list = []
    predicted_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            short_text_input, input_concepts, targets = batch.text[0], batch.concept[0], batch.label
            short_text_input, input_concepts, targets = short_text_input.to(device), input_concepts.to(device), targets.to(device)
            target_list.append(targets)

            output = classifier(short_text_input, input_concepts, network)
            predicted_list.append(output)

            loss = criterion(output, targets)
            test_loss += loss.item()

    predicted_list = torch.cat(predicted_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    accuracy, precision, recall, f1_score = metrics.assess(predicted_list, target_list)
    test_loss /= len(loader)

    return test_loss, accuracy, precision, recall, f1_score

def main():
    start_time = time.time()
    args = config.config()

    if not args.train_data_path:
        logger.info("please input train dataset path")
        exit()
    # if not (args.dev_data_path or args.test_data_path):
    #     logger.info("please input dev or test dataset path")
    #     exit()

    all_ = dataset.load_dataset(args.train_data_path, args.dev_data_path, args.test_data_path, \
                     args.txt_embedding_path, args.cpt_embedding_path, args.train_batch_size, \
                                                         args.dev_batch_size, args.test_batch_size)
    
    txt_TEXT, cpt_TEXT, txt_vocab_size, cpt_vocab_size, txt_word_embeddings, cpt_word_embeddings, \
           train_iter, dev_iter, test_iter, label_size = all_

    network = input("CNN or Linear: ")
    network = network.lower()

    parameters = input("Get Number of Parameters (y/n): ")
    parameters = parameters.lower()

    model = STCKAtten(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,\
                        cpt_word_embeddings, args.hidden_size, label_size, network)
    
    # if torch.cuda.is_available():
    model = model.to(device)
    
    train_data, test_data = dataset.train_test_split(train_iter, 0.8)
    train_data, dev_data = dataset.train_dev_split(train_data, 0.8)
    criterion = nn.CrossEntropyLoss()

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        test_loss, acc, p, r, f1 = test(model, test_data, loss_func, network)
        logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, acc, p, r, f1)
        return
    
    best_score = 0.0
    test_loss, test_acc, test_p, test_r, test_f1 = 0, 0, 0, 0, 0
    for epoch in range(args.epoch):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        train_loss, eval_loss, acc, p, r, f1 = train(model, train_data, dev_data, epoch, args.lr, optimizer, criterion, network)
        
        logger.info('Epoch:%d, Training Loss:%.4f', epoch, train_loss)
        logger.info('Epoch:%d, Eval Loss:%.4f, Eval Acc:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f', epoch, eval_loss, acc, p, r, f1)
        
        if f1 > best_score:
            best_score = f1
            torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))
            test_loss, test_acc, test_p, test_r, test_f1 = test(model, test_data, criterion, network)
        logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, test_acc, test_p, test_r, test_f1)

    if(parameters == "y"):
        print("The number of parameters are: ", len(list(model.parameters())))

    print("STCKA Runtime: %s seconds" % (time.time() - start_time))
    

if __name__ == "__main__":
    main()