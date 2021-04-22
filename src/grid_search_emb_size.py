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
    # args.emb_size
    args.epoch_count = 100
    # args.log_dir
    args.log_steps = False
    args.lr = 0.01
    args.mode = 'mean'
    # args.model
    # args.save_dir
    args.sent_len = 64
    args.test = True
    args.tokenizer = 'spacy'
    args.update_vectors = None
    args.vectors = None
    args.weight_factor = 1.0

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

    for dataset, sent_count in dataset_choices:
        for model in ['base', 'ower']:
            for emb_size in [1, 3, 10, 30, 100, 300, 1000]:
                args.ower_dir = f'data/ower/{dataset}'
                args.sent_count = sent_count

                args.emb_size = emb_size
                args.log_dir = f'runs/emb_size/{dataset}_{model}_{emb_size}'
                args.model = model
                args.save_dir = f'models/emb_size/{dataset}_{model}_{emb_size}'

                logging.info(f'Training on dataset {dataset} using model {model} with emb size {emb_size}')
                train(args)


if __name__ == '__main__':
    main()
