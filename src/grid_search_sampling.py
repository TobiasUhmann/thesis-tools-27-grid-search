import logging
from argparse import Namespace

from train import train


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = Namespace()

    # args.ower_dir
    args.class_count = 100
    # args.sent_count

    args.batch_size = 1024
    args.device = 'cuda'
    args.emb_size = None
    args.epoch_count = 20
    # args.log_dir
    args.log_steps = False
    args.lr = 0.01
    args.mode = 'mean'
    # args.model
    # args.save_dir
    args.sent_len = 64
    args.update_vectors = False
    args.vectors = 'glove.6B.300d'

    dataset_choices = [
        ('ower-v4-fb-irt-100-1-1', 1),
        ('ower-v4-fb-irt-100-1-2', 1),
        ('ower-v4-fb-irt-100-1-3', 1)
    ]

    for i in range(3):
        for model in ['base', 'ower']:
            for dataset, sent_count in dataset_choices:
                args.ower_dir = f'data/ower/{dataset}'
                args.sent_count = sent_count

                args.log_dir = f'runs/sampling_{model}_{dataset}_{i}'
                args.model = model
                args.save_dir = f'models/sampling_{model}_{dataset}_{i}'

                logging.info(f'Training model {model} on random-sample dataset {dataset}')
                train(args)


if __name__ == '__main__':
    main()
