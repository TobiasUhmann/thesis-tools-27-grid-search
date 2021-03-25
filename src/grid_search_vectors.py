import logging
from argparse import Namespace

from train import train


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = Namespace()

    args.ower_dir = 'data/ower/ower-v4-fb-irt-100-5/'
    args.class_count = 100
    args.sent_count = 5

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
    # args.vectors

    vectors_choices = [
        'charngram.100d',
        'fasttext.en.300d',
        'fasttext.simple.300d',
        'glove.42B.300d',
        'glove.840B.300d',
        'glove.twitter.27B.25d',
        'glove.twitter.27B.50d',
        'glove.twitter.27B.100d',
        'glove.twitter.27B.200d',
        'glove.6B.50d',
        'glove.6B.100d',
        'glove.6B.200d',
        'glove.6B.300d'
    ]

    for i in range(3):
        for model in ['base', 'ower']:
            for vectors in vectors_choices:

                args.log_dir = f'runs/vectors_{model}_{vectors}_{i}'
                args.model = model
                args.save_dir = f'models/vectors_{model}_{vectors}_{i}'
                args.vectors = vectors

                train(args)


if __name__ == '__main__':
    main()
