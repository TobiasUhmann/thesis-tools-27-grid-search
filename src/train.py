import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor, tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import Vocab
from tqdm import tqdm

from data.ower.ower_dir import OwerDir, Sample
from models.base import Base
from models.ower import Ower


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    train(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('ower_dir', metavar='ower-dir',
                        help='Path to (input) OWER Directory')

    parser.add_argument('class_count', metavar='class-count', type=int,
                        help='Number of classes distinguished by the classifier')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    device_choices = ['cpu', 'cuda']
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', choices=device_choices, default=default_device,
                        help='Where to perform tensor operations, one of {} (default: {})'.format(
                            device_choices, default_device))

    activation_choices = ['softmax', 'sigmoid', 'relu', 'none']
    default_activation = 'sigmoid'
    parser.add_argument('--activation', dest='activation', choices=activation_choices, default=default_activation,
                        help="Activation function for the OWER model's attention mechanism")

    default_batch_size = 1024
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='INT', default=default_batch_size,
                        help='Batch size (default: {})'.format(default_batch_size))

    default_emb_size = None
    parser.add_argument('--emb-size', dest='emb_size', type=int, metavar='INT', default=default_emb_size,
                        help='Word embedding size, --vectors will be ignored if set'
                             ' (default: {})'.format(default_emb_size))

    default_epoch_count = 20
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_log_dir = None
    parser.add_argument('--log-dir', dest='log_dir', metavar='STR', default=default_log_dir,
                        help='Tensorboard log directory (default: {})'.format(default_log_dir))

    parser.add_argument('--log-steps', dest='log_steps', action='store_true',
                        help='Log after steps, in addition to epochs')

    default_learning_rate = 0.01
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    mode_choices = ['mean', 'max', 'sum']
    default_mode = 'mean'
    parser.add_argument('--mode', dest='mode', choices=mode_choices, default=default_mode)

    model_choices = ['base', 'ower']
    default_model_choice = 'ower'
    parser.add_argument('--model', dest='model', choices=model_choices, default=default_model_choice,
                        help='Classifier to be trained (default: {})'.format(default_model_choice))

    default_save_dir = None
    parser.add_argument('--save-dir', dest='save_dir', metavar='STR', default=default_save_dir,
                        help='Model save directory (default: {})'.format(default_save_dir))

    default_sent_len = 64
    parser.add_argument('--sent-len', dest='sent_len', type=int, metavar='INT', default=default_sent_len,
                        help='Sentence length short sentences are padded and long sentences cropped to'
                             ' (default: {})'.format(default_sent_len))

    parser.add_argument('--test', dest='test', action='store_true',
                        help='Evaluate on test data after training')

    tokenizer_choices = ['spacy', 'whitespace']
    default_tokenizer = 'spacy'
    parser.add_argument('--tokenizer', dest='tokenizer', choices=tokenizer_choices, default=default_tokenizer,
                        help='How to split sentences into tokens (default: {})'.format(default_tokenizer))

    parser.add_argument('--update-vectors', dest='update_vectors', action='store_true',
                        help='Update pre-trained word embeddings during training')

    default_vectors = 'fasttext.simple.300d'
    parser.add_argument('--vectors', dest='vectors', metavar='STR', default=default_vectors,
                        help='Pre-trained word embeddings, ignored if --vectors is set'
                             ' (default: {})'.format(default_vectors))

    default_weight_factor = 1.0
    parser.add_argument('--weight-factor', dest='weight_factor', type=float, metavar='FLOAT',
                        default=default_weight_factor,
                        help="Factor by which to multiply the loss function's class weights,"
                             " no class weights applied if set to 0.0 (default: {})".format(default_weight_factor))

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dir', args.ower_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('    {:24} {}'.format('--activation', args.activation))
    logging.info('    {:24} {}'.format('--batch-size', args.batch_size))
    logging.info('    {:24} {}'.format('--device', args.device))
    logging.info('    {:24} {}'.format('--emb-size', args.emb_size))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--log-dir', args.log_dir))
    logging.info('    {:24} {}'.format('--log-steps', args.log_steps))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--mode', args.mode))
    logging.info('    {:24} {}'.format('--model', args.model))
    logging.info('    {:24} {}'.format('--save-dir', args.save_dir))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))
    logging.info('    {:24} {}'.format('--test', args.test))
    logging.info('    {:24} {}'.format('--tokenizer', args.tokenizer))
    logging.info('    {:24} {}'.format('--update-vectors', args.update_vectors))
    logging.info('    {:24} {}'.format('--vectors', args.vectors))

    return args


def train(args):
    ower_dir_path = args.ower_dir
    class_count = args.class_count
    sent_count = args.sent_count

    activation = args.activation
    batch_size = args.batch_size
    device = args.device
    emb_size = args.emb_size
    epoch_count = args.epoch_count
    log_dir = args.log_dir
    log_steps = args.log_steps
    lr = args.lr
    mode = args.mode
    model_name = args.model
    save_dir = args.save_dir
    sent_len = args.sent_len
    test = args.test
    tokenizer = args.tokenizer
    update_vectors = args.update_vectors
    vectors = args.vectors
    weight_factor = args.weight_factor

    #
    # Check that (input) OWER Directory exists
    #

    ower_dir = OwerDir(Path(ower_dir_path))
    ower_dir.check()

    #
    # Create (output) save dir if it does not exist already
    #

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    #
    # Load datasets
    #

    train_set: List[Sample]
    valid_set: List[Sample]
    test_set: List[Sample]

    if emb_size is not None:
        train_set, valid_set, test_set, vocab = ower_dir.read_datasets(class_count, sent_count, tokenizer=tokenizer)
    else:
        train_set, valid_set, test_set, vocab = ower_dir.read_datasets(class_count, sent_count, vectors, tokenizer)

    #
    # Create dataloaders
    #

    def generate_batch(batch: List[Sample]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param    batch:          [Sample(ent, [class], [sent])]

        :return:  ent_batch:      IntTensor[batch_size],
                  sents_batch:    IntTensor[batch_size, sent_count, sent_len],
                  classes_batch:  IntTensor[batch_size, class_count]
        """

        ent_batch, classes_batch, sents_batch = zip(*batch)

        for sents in sents_batch:
            shuffle(sents)

        cropped_sents_batch = [[sent[:sent_len] for sent in sents] for sents in sents_batch]
        padded_sents_batch = [[sent + [0] * (sent_len - len(sent)) for sent in sents] for sents in cropped_sents_batch]

        return tensor(ent_batch), tensor(padded_sents_batch), tensor(classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=generate_batch)

    #
    # Calc class weights
    #

    _, train_classes_stack, _ = zip(*train_set)
    train_freqs = np.array(train_classes_stack).mean(axis=0)

    if weight_factor == 0.0:
        class_weights = None
    else:
        class_weights = tensor(1 / train_freqs * weight_factor)

    #
    # Create model
    #

    model = create_model(model_name, emb_size, vocab, class_count, mode, update_vectors, activation).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    writer = SummaryWriter(log_dir=log_dir)

    #
    # Train and validate
    #

    best_valid_f1 = 0

    # Global progress for Tensorboard
    train_progress = 0
    valid_progress = 0

    for epoch in range(epoch_count):

        epoch_metrics = {
            'train': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []},
            'valid': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []}
        }

        #
        # Train
        #

        for _, sents_batch, gt_batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            train_progress += len(sents_batch)

            sents_batch = sents_batch.to(device)
            gt_batch = gt_batch.to(device).float()

            logits_batch = model(sents_batch)
            loss = criterion(logits_batch, gt_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #
            # Log metrics
            #

            pred_batch = (logits_batch > 0).int()

            step_loss = loss.item()
            step_pred_batch = pred_batch.cpu().numpy().tolist()
            step_gt_batch = gt_batch.cpu().numpy().tolist()

            epoch_metrics['train']['loss'] += step_loss
            epoch_metrics['train']['pred_classes_stack'] += step_pred_batch
            epoch_metrics['train']['gt_classes_stack'] += step_gt_batch

            if log_steps:
                writer.add_scalars('loss', {'train': step_loss}, train_progress)

                step_metrics = {'train': {
                    'pred_classes_stack': step_pred_batch,
                    'gt_classes_stack': step_gt_batch
                }}

                log_class_metrics(step_metrics, writer, train_progress, class_count)
                log_macro_metrics(step_metrics, writer, train_progress)

        #
        # Validate
        #

        with torch.no_grad():
            for _, sents_batch, gt_batch, in tqdm(valid_loader, desc=f'Epoch {epoch}'):
                valid_progress += len(sents_batch)

                sents_batch = sents_batch.to(device)
                gt_batch = gt_batch.to(device).float()

                logits_batch = model(sents_batch)
                loss = criterion(logits_batch, gt_batch)

                #
                # Log metrics
                #

                pred_batch = (logits_batch > 0).int()

                step_loss = loss.item()
                step_pred_batch = pred_batch.cpu().numpy().tolist()
                step_gt_batch = gt_batch.cpu().numpy().tolist()

                epoch_metrics['valid']['loss'] += step_loss
                epoch_metrics['valid']['pred_classes_stack'] += step_pred_batch
                epoch_metrics['valid']['gt_classes_stack'] += step_gt_batch

                if log_steps:
                    writer.add_scalars('loss', {'valid': step_loss}, valid_progress)

                    step_metrics = {'valid': {
                        'pred_classes_stack': step_pred_batch,
                        'gt_classes_stack': step_gt_batch
                    }}

                    log_class_metrics(step_metrics, writer, valid_progress, class_count)
                    log_macro_metrics(step_metrics, writer, valid_progress)

        #
        # Log loss
        #

        train_loss = epoch_metrics['train']['loss'] / len(train_loader)
        valid_loss = epoch_metrics['valid']['loss'] / len(valid_loader)

        writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)

        #
        # Log metrics
        #

        log_class_metrics(epoch_metrics, writer, epoch, class_count)
        valid_f1 = log_macro_metrics(epoch_metrics, writer, epoch)

        #
        # Store model
        #

        if (save_dir is not None) and (valid_f1 > best_valid_f1):
            best_valid_f1 = valid_f1

            with open(f'{save_dir}/model.pt', 'wb') as f:
                pickle.dump(model, f)

    #
    # Load model and test
    #

    if test:
        if save_dir is not None:
            with open(f'{save_dir}/model.pt', 'rb') as f:
                model = pickle.load(f)

        test_progress = 0
        epoch_metrics = {
            'test': {'loss': 0.0, 'pred_classes_stack': [], 'gt_classes_stack': []}
        }

        with torch.no_grad():
            for _, sents_batch, gt_batch, in tqdm(test_loader, desc=f'Test'):
                test_progress += len(sents_batch)

                sents_batch = sents_batch.to(device)
                gt_batch = gt_batch.to(device).float()

                logits_batch = model(sents_batch)
                loss = criterion(logits_batch, gt_batch)

                #
                # Log metrics
                #

                pred_batch = (logits_batch > 0).int()

                step_loss = loss.item()
                step_pred_batch = pred_batch.cpu().numpy().tolist()
                step_gt_batch = gt_batch.cpu().numpy().tolist()

                epoch_metrics['test']['loss'] += step_loss
                epoch_metrics['test']['pred_classes_stack'] += step_pred_batch
                epoch_metrics['test']['gt_classes_stack'] += step_gt_batch

                if log_steps:
                    writer.add_scalars('loss', {'test': step_loss}, test_progress)

                    step_metrics = {'test': {
                        'pred_classes_stack': step_pred_batch,
                        'gt_classes_stack': step_gt_batch
                    }}

                    log_class_metrics(step_metrics, None, test_progress, class_count)
                    log_macro_metrics(step_metrics, None, test_progress)

        #
        # Log loss
        #

        test_loss = epoch_metrics['test']['loss'] / len(test_loader)
        logging.info(f'Test Loss = {test_loss}')

        #
        # Log metrics
        #

        log_class_metrics(epoch_metrics, None, -1, class_count)

        for split, metrics in epoch_metrics.items():
            prec, rec, f1, _ = precision_recall_fscore_support(metrics['gt_classes_stack'],
                                                               metrics['pred_classes_stack'],
                                                               average='macro',
                                                               zero_division=0)

            with open(f'{save_dir}/eval.txt', 'w') as f:
                f.write(f'Precision = {prec:.4f}\n')
                f.write(f'Recall = {rec:.4f}\n')
                f.write(f'F1 = {f1:.4f}\n')


def create_model(model_name: str, emb_size: int, vocab: Vocab, class_count: int, mode: str, update_vectors: bool,
                 activation: str):
    if model_name == 'base':
        if emb_size is None:
            return Base.from_pre_trained(vocab, class_count, mode, update_vectors)
        else:
            return Base.from_random(len(vocab), emb_size, class_count, mode)

    elif model_name == 'ower':
        if emb_size is None:
            return Ower.from_pre_trained(vocab, class_count, mode, update_vectors, activation)
        else:
            return Ower.from_random(len(vocab), emb_size, class_count, mode, activation)

    else:
        raise


def log_class_metrics(data: Dict, writer: Optional[SummaryWriter], x: int, class_count: int) -> None:
    """
    Calculate class-wise metrics and log metrics of most/least common metrics to Tensorboard

    :param data: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                  'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}

    :param class_count: Log <class_count> most common and <class_count> least common classes
    """

    most_common_classes = range(0, 3)
    least_common_classes = range(class_count - 3, class_count)
    log_classes = [*most_common_classes, *least_common_classes]

    for split, metrics in data.items():
        prfs_list = precision_recall_fscore_support(metrics['gt_classes_stack'],
                                                    metrics['pred_classes_stack'],
                                                    average=None,
                                                    zero_division=0)

        # c = class
        for c, (prec, rec, f1, supp), in enumerate(zip(*prfs_list)):
            if c not in log_classes:
                continue

            if writer is None:
                logging.info(f'Precision {c} = {prec:.4f}')
                logging.info(f'Recall {c} = {rec:.4f}')
                logging.info(f'F1 {c} = {f1:.4f}')
                logging.info(f'Support {c} = {supp}')
            else:
                writer.add_scalars('precision', {f'{split}_{c}': prec}, x)
                writer.add_scalars('recall', {f'{split}_{c}': rec}, x)
                writer.add_scalars('f1', {f'{split}_{c}': f1}, x)
                writer.add_scalars('support', {f'{split}_{c}': supp}, x)


def log_macro_metrics(data: Dict, writer: Optional[SummaryWriter], x: int) -> float:
    """
    Calculate macro metrics across all classes and log them to Tensorboard

    :param data: {'train': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]},
                  'valid': {'gt_classes_stack': [[gt_class]], 'pred_classes_stack': [[pred_class]]}}

    :param x: Value on x-axis

    :return F1 score
    """

    for split, metrics in data.items():
        prec, rec, f1, _ = precision_recall_fscore_support(metrics['gt_classes_stack'],
                                                           metrics['pred_classes_stack'],
                                                           average='macro',
                                                           zero_division=0)

        if writer is None:
            logging.info(f'Precision = {prec:.4f}')
            logging.info(f'Recall = {rec:.4f}')
            logging.info(f'F1 = {f1:.4f}')
        else:
            writer.add_scalars('precision', {split: prec}, x)
            writer.add_scalars('recall', {split: rec}, x)
            writer.add_scalars('f1', {split: f1}, x)

    return f1


if __name__ == '__main__':
    main()
