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
    # args.emb_size
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

    for i in range(3):
        for model in ['base', 'ower']:
            for emb_size in [1, 3, 5, 10, 15, 25, 50, 100, 200, 300, 400]:

                args.emb_size = emb_size
                args.log_dir = f'runs/emb_size_{model}_{emb_size}_{i}'
                args.model = model
                args.save_dir = f'models/emb_size_{model}_{emb_size}_{i}'

                logging.info(f'Training model {model} with emb size {emb_size}')
                train(args)


if __name__ == '__main__':
    main()
