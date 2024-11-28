import argparse
import os

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    trainer = Trainer(args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, args.train_dir)
        dev_dataset = load_and_cache_examples(args, tokenizer, args.eval_dir)
        trainer.train(train_dataset, dev_dataset)

    if args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, args.test_dir)
        trainer.load_model()
        trainer.evaluate(test_dataset, args.test_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
    parser.add_argument("--checkpoint", default=-1, type=str, help="Checkpoint to load, -1 or unset to load the latest checkpoint")

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--train_dir", help="Directory with training data")
    parser.add_argument("--eval_dir", help="Directory with evaluation data")
    parser.add_argument("--test_dir", help="Directory with test data, used with do_eval")

    parser.add_argument("--ignore_index", default=-100, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    args = parser.parse_args()
    if not args.train_dir:
        args.train_dir = os.path.join(args.data_dir, args.task, "train")
    if not args.eval_dir:
        args.eval_dir = os.path.join(args.data_dir, args.task, "dev")
    if not args.test_dir:
        args.test_dir = os.path.join(args.data_dir, args.task, "test")
    main(args)
