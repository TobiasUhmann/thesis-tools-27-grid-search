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
    args.lr = 0.003
    args.mode = 'mean'
    # args.model
    args.optim = 'adam'
    # args.save_dir
    args.sent_len = 64
    args.test = True
    args.tokenizer = 'spacy'
    args.update_vectors = True
    args.vectors = 'fasttext.simple.300d'
    # args.weight_factor

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
            for weight_factor in [0, 0.5, 1.0, 2.0]:
                args.ower_dir = f'data/ower/{dataset}'
                args.sent_count = sent_count

                args.log_dir = f'runs/weight_factor/{dataset}_{model}_{weight_factor}'
                args.model = model
                args.save_dir = f'models/weight_factor/{dataset}_{model}_{weight_factor}'
                args.weight_factor = weight_factor

                logging.info(f'Training on dataset {dataset} using model {model} with weight factor {weight_factor}')
                train(args)


if __name__ == '__main__':
    main()
