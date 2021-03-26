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
    args.epoch_count = 50
    # args.log_dir
    args.log_steps = False
    args.lr = 0.01
    args.mode = 'mean'
    # args.model
    # args.save_dir
    args.sent_len = 64
    # args.update_vectors
    args.vectors = 'glove.6B.300d'

    for i in range(3):
        for model in ['base', 'ower']:
            for update_vectors in [False, True]:

                args.log_dir = f'runs/update_vectors_{model}_{update_vectors}_{i}'
                args.model = model
                args.save_dir = f'models/update_vectors_{model}_{update_vectors}_{i}'
                args.update_vectors = update_vectors

                logging.info(f'Training model {model} with --update-vectors == {update_vectors}')
                train(args)


if __name__ == '__main__':
    main()
