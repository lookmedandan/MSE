from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from evaluate import eval_sentence, cws_evaluate_word_PRF, cws_evaluate_OOV, get_NER_scores
import datetime
import pickle as pkl
from models import CWSB
from utils import Config, Process
import pickle
from loss_function import CrossEntropyBoundSmoothLoss_ScaleAverage,CrossEntropyBoundSmoothLoss

torch.set_printoptions(profile='full')
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_num = 1

def train(args):

    global log_num
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)
    
    logger.info(vars(args))

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {}, 16-bits training: {}".format(device, args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    os.environ['PYTHONHASHSEED'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    # Get a dictionary for the CWS task.
    if args.dict_path is not None:
        word2id = {'<PAD>': 0} 
        with open('./data/{}/train.tsv'.format(args.data_name), 'r', encoding='utf8') as ins:
            word = ''
            for line in ins:
                splits = line.strip().split('\t')
                character = splits[0]
                label = splits[-1]
                word += character
                if label in ['S', 'E']:
                    if word not in word2id:
                        word2id[word] = len(word2id)
                    word = ''
        with open(args.dict_path, 'wb') as f:
            pkl.dump(word2id, f)
        logger.info('# of character in train: %d' % len(word2id))
    else:
        word2id = None

    config = Config(args)
    process = Process(config)

    tmodel = None
    smodel = CWSB(config.cache_dir, config.bert_model, config.num_labels, args.use_crf)

    total_params = sum(p.numel() for p in smodel.parameters() if p.requires_grad)
    logger.info('Model name: %s\n\n' % args.output_model_dir)
    logger.info('# of trainable parameters: %d\n' % total_params)

    logger.info("***** Running training *****")
    
    # datas
    if word2id is None and args.data_name in ['weibo', 'resume', 'msra']:  # ner task.
        if os.path.exists('./train.pickle')==True:
            with open('./train.pickle', 'rb') as file:
                train_examples = pickle.load(file)
        else:
            if args.read_drop == True:
                train_examples = process.read_chinese_ner_data \
                    ('../data/{}/{}.txt'.format(args.data_name, 'train'), read_drop = args.read_drop, shuffle_dict = args.shuffle_dict)
                file = open('./train.pickle', 'wb')
                pickle.dump(train_examples, file)
                file.close()
                print(train_examples[:10])
    
        train_examples = train_examples[:int(len(train_examples)*args.ratio)]
        if os.path.exists('./dev.pickle')==True:
            with open('./dev.pickle', 'rb') as file:
                eval_examples = pickle.load(file)
        else:
            eval_examples = process.read_chinese_ner_data \
                ('../data/{}/{}.txt'.format(args.data_name, 'dev'), 'dev')
            file = open('./dev.pickle', 'wb')
            pickle.dump(eval_examples, file)
            file.close()
    else:  # cws task
        all_train_examples = process.load_data('./data/{}/{}.tsv'.format(args.data_name, 'train'))
        np.random.shuffle(all_train_examples)
        all_train_examples = all_train_examples[:int(len(all_train_examples)*args.ratio)]
        train_num = int(len(all_train_examples)*args.train_ratio)
        logger.info("\nRatio: %f Train size: %d Test size: %d", args.ratio, train_num, len(all_train_examples)-train_num)
        train_examples = all_train_examples[:train_num]
        eval_examples = all_train_examples[train_num:]

    if os.path.exists('./test.pickle')==True:
        with open('./test.pickle', 'rb') as file:
            test_examples = pickle.load(file)
    else:
        test_examples = process.read_chinese_ner_data \
            ('../data/{}/{}.txt'.format(args.data_name, 'test'), 'test')
        file = open('./test.pickle', 'wb')
        pickle.dump(test_examples, file)
        file.close()

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.fp16:
        smodel.half()
    smodel.to(device)

    param_optimizer = list(smodel.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=False)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    history = {'epoch': [], 'p': [], 'r': [], 'f': []}
    test_history = {'epoch': [], 'p': [], 'r': [], 'f': []}
    loss_history = {'epoch': [], 'batch': [], 'loss':[], 'entity_token_count':[], 'context_token_count':[], 'entity_weight':[], 'context_weight':[]}
    num_of_no_improvement = 0
    patient = args.patient
    global_step = 0
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    mse_loss = torch.nn.MSELoss()
    with open('../data/{}/tag.json'.format(args.data_name),'r', encoding='utf-8')as f:
        label2id = json.load(f)
    bound_ids = []
    for l,i in label2id.items():
        if 'B-' in l or 'S-' in l or 'E-' in l:
            bound_ids.append(i)
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        np.random.shuffle(train_examples)
        # Train
        for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
            smodel.train()
            batch_examples = train_examples[start_index: min(start_index +
                                                                args.train_batch_size, len(train_examples))]
            if len(batch_examples) == 0:
                continue

            train_features = process.convert_examples_to_features(batch_examples, args.entity_weight, args.context_weight, args.use_adaptive_weight)
            input_ids, label_ids, input_mask, weights, entity_token_count, context_token_count = process.feature2input(device, train_features)
            
            soutput, slogits = smodel(input_ids)

            if smodel.crf is None:
                bound_loss = CrossEntropyBoundSmoothLoss_ScaleAverage(args.e, args.D, bound_ids, slogits.size(0), slogits.size(1), slogits.size(2), device)
                loss = bound_loss(slogits.view(-1, config.num_labels), label_ids.view(-1))
            else:
                loss = smodel.crf.neg_log_likelihood_loss(slogits, input_mask, label_ids)
            
            loss_history['loss'].append(loss.item())
            loss_history['epoch'].append(epoch)
            loss_history['batch'].append(step)
            loss_history['entity_token_count'].append(entity_token_count.to('cpu').numpy())
            loss_history['context_token_count'].append(context_token_count.to('cpu').numpy())
            loss_history['entity_weight'].append(args.entity_weight)
            loss_history['context_weight'].append(args.context_weight)

            if tmodel is not None:
                toutput, tlogits = tmodel(input_ids)
                toutput, tlogits = toutput.detach(), tlogits.detach()
                if tmodel.crf is None:
                    tlabel_preds = torch.argmax(F.log_softmax(tlogits, dim=2), dim=2)
                else:
                    _, tlabel_preds = tmodel.crf._viterbi_decode(tlogits, input_mask)
                t_eta = torch.abs(tlabel_preds - label_ids).clamp(0, 1)
                t_eta = 1 - t_eta

                if smodel.crf is None:
                    slabel_preds = torch.argmax(F.log_softmax(slogits, dim=2), dim=2)
                else:
                    _, slabel_preds = smodel.crf._viterbi_decode(slogits, input_mask)
                s_eta = torch.abs(slabel_preds - label_ids).clamp(0, 1)
                s_eta = 1 - s_eta

                if args.num_wei == 1:
                    w_wei = t_eta + s_eta + 1 # Weight one
                elif args.num_wei == 2:
                    w_wei = 2 * t_eta + s_eta + 1 # Weight two
                elif args.num_wei == 3:
                    w_wei = t_eta + 2 * s_eta + 1 # Weight three
                elif args.num_wei == 4:
                    w_wei = 2 * t_eta - s_eta + 2  # Weight four
                else:
                    w_wei = 1  # No weight
                if not args.distill_hidden:
                    weights = weights.unsqueeze(2)
                    if args.num_wei>=1:
                        w_wei = w_wei.unsqueeze(2)
                    loss = (1 - args.alpha) * loss + args.alpha * mse_loss(weights * w_wei * slogits, weights * w_wei * tlogits) # targets
                else:
                    weights = weights.unsqueeze(2)
                    if args.num_wei>=1:
                        w_wei = w_wei.unsqueeze(2)
                    loss = (1 - args.alpha) * loss + args.alpha * mse_loss(weights * w_wei * soutput, weights * w_wei * toutput)  # hiddens

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                        args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        # Dev
        if word2id is None:
            p, r, f = dev_ner(config, smodel, process, args.eval_batch_size, eval_examples)  # ner
        else:
            p, r, f, _ = dev(config, smodel, word2id, process, args.eval_batch_size, eval_examples)  # cws
        history['epoch'].append(epoch)
        history['p'].append(p)
        history['r'].append(r)
        history['f'].append(f)
        logger.info("\nEpoch: %d, P: %f, R: %f, F: %f", epoch + 1, p, r, f)
        logger.info("=======entity level========")

        temp_f = f - best_f
        if temp_f >= 0.0001:
            logger.info(temp_f)
            best_epoch = epoch + 1
            best_p = p
            best_r = r
            best_f = f
            num_of_no_improvement = 0

            if not os.path.exists(args.output_model_dir):
                os.makedirs(args.output_model_dir)

            if config.output_model_dir:
                torch.save(smodel, os.path.join(config.output_model_dir, 'model.pt'))

        else:
            if epoch>=6:
                num_of_no_improvement += 1
        
        # smodel is our testing target with the best performance
        logger.info("*** Testing ***")
        if word2id is None:
            tp, tr, tf = dev_ner(config, smodel, process, args.eval_batch_size, test_examples)  # ner
            logger.info("P: %f, R: %f, F: %f", tp, tr, tf)
            logger.info("P: %f, R: %f, F: %f", tp, tr, tf)
            test_history['epoch'].append(epoch)
            test_history['p'].append(tp)
            test_history['r'].append(tr)
            test_history['f'].append(tf)
        else:
            tp, tr, tf, toov = dev(config, smodel, word2id, process, args.eval_batch_size, test_examples) # cws
            logger.info("P: %f, R: %f, F: %f, OOV: %f", tp, tr, tf, toov)
        logger.info("*** Testing ***")

        # teacher best
        if args.num_wei >= 0 and num_of_no_improvement == 0 and not args.use_last:
            tmodel = torch.load(os.path.join(config.output_model_dir, 'model.pt'))
            if args.fp16:
                tmodel.half()
            tmodel.to(device).eval()
        
        # teacher last
        if args.num_wei >= 0 and args.use_last:
            if not os.path.exists(args.output_model_dir):
                os.makedirs(args.output_model_dir)
            if config.output_model_dir:
                torch.save(smodel, os.path.join(config.output_model_dir, 'teacher.pt'))
            tmodel = torch.load(os.path.join(config.output_model_dir, 'teacher.pt'))
            if args.fp16:
                tmodel.half()
            tmodel.to(device).eval()
        
        if best_f > 0 and num_of_no_improvement >= patient:
            logger.info('\nEarly stop triggered at epoch %d\n' % (epoch+1))
            break

    # End of training
    logger.info("***  Best Epoch: %d ***", best_epoch)
    if os.path.exists(config.output_model_dir):
        file_path = os.path.join(config.output_model_dir, 'model.json')
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(history, f)
            f.write('\n')
    
    df = pd.DataFrame()
    df['epoch'] = history['epoch']
    df['p'] = history['p']
    df['r'] = history['r']
    df['f'] = history['f']
    df.to_csv(f'./dev_{log_num}.csv', encoding='utf-8')
    df = pd.DataFrame()
    df['epoch'] = test_history['epoch']
    df['p'] = test_history['p']
    df['r'] = test_history['r']
    df['f'] = test_history['f']
    df.to_csv(f'./test_{log_num}.csv', encoding='utf-8')
    df = pd.DataFrame()
    df['epoch'] = loss_history['epoch']
    df['batch'] = loss_history['batch']
    df['loss'] = loss_history['loss']
    df['entity_token_count'] = loss_history['entity_token_count']
    df['context_token_count'] = loss_history['context_token_count']
    df['entity_weight'] = loss_history['entity_weight']
    df['context_weight'] = loss_history['context_weight']
    df.to_csv(f'./loss_{log_num}.csv', encoding='utf-8')
    log_num = log_num + 1


def dev(config, model, word2id, process, eval_batch_size, eval_examples):
    model.eval()
    y_true = []
    y_pred = []

    for start_index in range(0, len(eval_examples), eval_batch_size):
        eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                len(eval_examples))]
        eval_features = process.convert_examples_to_features(eval_batch_examples)  
        input_ids, label_ids, input_mask = process.feature2input(device, eval_features)

        with torch.no_grad():
            _, logits = model(input_ids)
            if model.crf is None:
                tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            else:
                _, tag_seq = model.crf._viterbi_decode(logits, input_mask)

        label_ids = label_ids.to('cpu').numpy()
        label_preds = tag_seq.to('cpu').numpy()
        assert(len(label_ids) == len(label_preds)), 'NOT EQUAL!'
        for label_id, label_pred, input_mas in zip(label_ids, label_preds, input_mask):
            tmpt = []
            tmpp = []
            for iid, pid, imas in zip(label_id, label_pred, input_mas):
                if imas == 0:
                    break
                tmpt.append(config.id2label[iid])
                tmpp.append(config.id2label.get(pid, 'E'))
            if len(tmpt) < 2:
                continue
            y_true.append(tmpt[1:-1])
            y_pred.append(tmpp[1:-1])

    y_true_all = []
    y_pred_all = []
    sentence_all = []
    for y_true_item, y_pred_item, eval_example in zip(y_true, y_pred, eval_examples):
        y_true_all += y_true_item
        y_pred_all += y_pred_item
        sen = eval_example[0]
        if len(y_true_item) != len(sen):
            sen = sen[:len(y_true_item)]
        sentence_all.append(sen)
    
    p, r, f = cws_evaluate_word_PRF(y_pred_all, y_true_all)
    oov = cws_evaluate_OOV(y_pred, y_true, sentence_all, word2id)
    return p, r, f, oov


def dev_ner(config, model, process, eval_batch_size, eval_examples):
    model.eval()
    y_true = []
    y_pred = []

    for start_index in range(0, len(eval_examples), eval_batch_size):
        eval_batch_examples = eval_examples[start_index: min(start_index + eval_batch_size,
                                                                len(eval_examples))]
        eval_features = process.convert_examples_to_features(eval_batch_examples, args.entity_weight, args.context_weight, args.use_adaptive_weight)  
        input_ids, label_ids, input_mask, weights, entity_token_count, context_token_count = process.feature2input(device, eval_features)

        with torch.no_grad():
            _, logits = model(input_ids)
            if model.crf is None:
                tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            else:
                _, tag_seq = model.crf._viterbi_decode(logits, input_mask)

        label_ids = label_ids.to('cpu').numpy()
        label_preds = tag_seq.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        assert(len(label_ids) == len(label_preds)), 'NOT EQUAL!'
        for label_id, label_pred, input_mas in zip(label_ids, label_preds, input_mask):
            tmpt = []
            tmpp = []
            for iid, pid, imas in zip(label_id, label_pred, input_mas):
                if imas == 0:
                    break
                tmpt.append(config.id2label.get(iid, config.num_labels))
                tmpp.append(config.id2label.get(pid, config.id2label[1]))
            if len(tmpt) < 2:
                continue
            y_true.append(tmpt[1:-1])
            y_pred.append(tmpp[1:-1])

    p, r, f, nums = get_NER_scores(y_true, y_pred)

    logger.info("gold_num = {}\t pred_num = {}\t right_num = {}".format(nums[0], nums[1], nums[2]))
    
    return p, r, f


def test(args):
    logger.info(vars(args))

    model = torch.load(args.test_model)
    with open(args.dict_path, 'rb') as f:
        word2id = pkl.load(f)
    config = Config(args)
    process = Process(config)
    eval_examples = process.load_data('./data/{}/{}.tsv'.format(args.data_name, 'test'))
    if args.fp16:
        model.half()
    model.to(device)
    model.eval()
    # verify
    p, r, f, oov = dev(config, model, word2id, process, args.eval_batch_size, eval_examples)

    print("\nTesting P: %f, R: %f, F: %f, OOV: %f\n" % (p, r, f, oov))
    logger.info("\n=======  Testing  ========")
    logger.info("\nP: %f, R: %f, F: %f, OOV: %f\n", p, r, f, oov)
    logger.info("\n=======  Testing  ========")


def predict(args):
    model = torch.load(args.test_model)
    config = Config(args)
    process = Process(config)
    eval_examples = process.load_data(args.input_file, do_predict=True)
    if args.fp16:
        model.half()
    model.to(device)
    model.eval()
    y_pred = []

    for start_index in range(0, len(eval_examples), args.eval_batch_size):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                len(eval_examples))]
        eval_features = process.convert_examples_to_features(eval_batch_examples)
        input_ids, label_ids, input_mask = process.feature2input(device, eval_features)
        with torch.no_grad():
            _, logits = model(input_ids)
            if model.crf is None:
                tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            else:
                _, tag_seq = model.crf._viterbi_decode(logits, input_mask)

        label_ids = label_ids.to('cpu').numpy()
        label_preds = tag_seq.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp = []
            for j, _ in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == config.num_labels - 1:
                    y_pred.append(temp)
                    break
                else:
                    temp.append(config.id2label[label_preds[i][j]])

    print('write results to %s' % str(args.output_file))
    with open(args.output_file, 'w', encoding='utf8') as writer:
        for i in range(len(y_pred)):
            seg_pred_str = eval_sentence(y_pred[i], eval_examples[i][0])
            writer.write('%s\n' % seg_pred_str)
    print('===Finished!!!===')


def main(args):
    if args.do_train:
        train(args)
    elif args.do_test:  # only for cws tasks.
        test(args)
    elif args.do_predict:  # only for cws tasks.
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_predict` must be True.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_crf",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_last",
                        action='store_true',
                        help="take last model as a teacher.")
    parser.add_argument("--alpha",
                        default=0.3,
                        type=float,
                        help="The sample ratio.")
    parser.add_argument("--train_ratio",
                        default=0.9,
                        type=float,
                        help="The train ratio.")
    parser.add_argument("--num_wei",
                        default=-1,
                        type=int,
                        help="-1: no distill; 0: no weight; 1: weight one; 2: Weight two...")
    parser.add_argument("--distill_hidden",
                        action='store_true',
                        help="take last model as a teacher.")                        
    parser.add_argument("--data_name",
                        default='pku',
                        type=str,
                        help="CWS: pku, msr, cityu, as; NER: weibo, ontonote4, resume, msra.")
    parser.add_argument("--dict_path",
                        default=None,
                        type=str,
                        help="The dict path of corpus.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--test_model", default=None, type=str,
                        help="")
    parser.add_argument('--model_name', type=str, default=None, help="")
    
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument("--max_seq_length",
                        default=300,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--ratio",
                        default=1.0,
                        type=float,
                        help="The sample ratio.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--bert_model", default='./pretrained_models/bert-base-chinese/', type=str,
                        help="bert-base-uncased or roberta-base-zh.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--output_model_dir', type=str, default=None, help="")
    parser.add_argument('--save_top',
                        type=int,
                        default=1,
                        help="")
    parser.add_argument('--read_drop', action='store_true', help='')
    parser.add_argument('--shuffle_dict', action='store_true', help='shuffle entity dict.')
    parser.add_argument('--e',type=float,default=0.1,help='smooth ratio.')
    parser.add_argument('--D',type=int, default=2, help='smooth distance.')
    parser.add_argument('--entity_weight',type=float, default=2.0)
    parser.add_argument('--context_weight',type=float, default=1.0)
    parser.add_argument("--use_adaptive_weight", action='store_true')
    args = parser.parse_args()
    args.do_train = True
    args.do_test = False
    args.do_predict = False
    args.use_crf = False
    args.use_last = False
    args.model_name = 'weibo'
    args.data_name = 'weibo'
    args.patient = 8
    args.seed = 42
    args.train_batch_size = 8
    args.read_drop = True    
    args.shuffle_dict = False
    args.e = 0.20
    args.D = 1
    args.entity_weight = 2.0
    args.num_train_epochs = 20
    args.use_adaptive_weight = True
    args.bert_model = "../PretrainedModel/bert-base-chinese"
    args.distill_hidden = True
    for n in [0,1,2,3,4]:
        args.num_wei = n
        torch.cuda.empty_cache()
        main(args)
    args.distill_hidden = False
    for n in [0,1,2,3,4]:
        args.num_wei = n
        torch.cuda.empty_cache()
        main(args)