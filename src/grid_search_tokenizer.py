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
    args.epoch_count = 100
    # args.log_dir
    args.log_steps = False
    args.lr = 0.01
    args.mode = 'mean'
    # args.model
    # args.save_dir
    args.sent_len = 64
    # args.tokenizer
    args.update_vectors = False
    args.vectors = 'glove.6B.300d'

    dataset_choices = [
        ('ower-v4-cde-cde-100-1', 1),
        ('ower-v4-cde-irt-100-1', 1),
        ('ower-v4-cde-irt-100-5', 5),
        ('ower-v4-cde-irt-100-15', 15),
        ('ower-v4-cde-irt-100-30', 20),
        ('ower-v4-fb-irt-100-1', 1),
        ('ower-v4-fb-irt-100-5', 5),
        ('ower-v4-fb-irt-100-15', 15),
        ('ower-v4-fb-irt-100-30', 30),
        ('ower-v4-fb-owe-100-1', 1)
    ]

    for i in range(3):
        for dataset, sent_count in dataset_choices:
            for model in ['base', 'ower']:
                for tokenizer in ['spacy', 'whitespace']:

                    args.ower_dir = f'data/ower/{dataset}'
                    args.sent_count = sent_count

                    args.log_dir = f'runs/tokenizers_{model}_{dataset}_{tokenizer}_{i}'
                    args.model = model
                    args.save_dir = f'models/tokenizers_{model}_{dataset}_{tokenizer}_{i}'
                    args.tokenizer = tokenizer

                    logging.info(f'Training model {model} on dataset {dataset} with tokenizer {tokenizer}')
                    train(args)


if __name__ == '__main__':
    main()