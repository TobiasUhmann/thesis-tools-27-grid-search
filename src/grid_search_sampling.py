import logging
from argparse import Namespace

from train import train


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = Namespace()

    # args.ower_dir
    args.class_count = 100
    # args.sent_count

    args.activation = 'sigmoid'
    args.batch_size = 1024
    args.device = 'cuda'
    args.emb_size = None
    args.epoch_count = 100
    # args.log_dir
    args.log_steps = False
    args.lr = 0.01
    args.mode = 'mean'
    # args.model
    args.optim = 'adam'
    # args.save_dir
    args.sent_len = 64
    args.test = True
    args.tokenizer = 'spacy'
    args.update_vectors = True
    args.vectors = 'fasttext.simple.300d'
    args.weight_factor = 1.0

    dataset_choices = [
        ('ower-v4-fb-irt-100-1-1', 1),
        ('ower-v4-fb-irt-100-1-2', 1),
        ('ower-v4-fb-irt-100-1-3', 1)
    ]

    for dataset, sent_count in dataset_choices:
        for model in ['base', 'ower']:
            args.ower_dir = f'data/ower/{dataset}'
            args.sent_count = sent_count

            args.log_dir = f'runs/sampling/{dataset}_{model}'
            args.model = model
            args.save_dir = f'models/sampling/{dataset}_{model}'

            logging.info(f'Training on dataset {dataset} using model {model}')
            train(args)


if __name__ == '__main__':
    main()
