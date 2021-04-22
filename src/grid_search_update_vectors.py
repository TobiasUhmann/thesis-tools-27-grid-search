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
    args.epoch_count = 200
    # args.log_dir
    args.log_steps = False
    args.lr = 0.01
    args.mode = 'mean'
    # args.model
    # args.save_dir
    args.sent_len = 64
    args.test = True
    args.tokenizer = 'spacy'
    # args.update_vectors
    # args.vectors
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
            for vectors in ['charngram.100d', 'fasttext.simple.300d', 'glove.6B.300d']:
                for update_vectors in [False, True]:
                    args.ower_dir = f'data/ower/{dataset}'
                    args.sent_count = sent_count

                    args.log_dir = f'runs/update/{dataset}_{model}_{vectors}_{update_vectors}'
                    args.model = model
                    args.save_dir = f'models/update/{dataset}_{model}_{vectors}_{update_vectors}'
                    args.update_vectors = update_vectors
                    args.vectors = vectors

                    logging.info(f'Training on dataset {dataset} using model {model}'
                                 f' with vectors {vectors} and update vectors {update_vectors}')
                    train(args)


if __name__ == '__main__':
    main()
