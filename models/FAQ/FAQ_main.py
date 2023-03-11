from .FAQ_model import FAQ
from utils import merge_arg_and_config, convert_config_paths
from config import faq_model_config, faq_runner_config
import tensorflow as tf
import sys
import argparse

def main():
    model = FAQ()
    parser = argparse.ArgumentParser(usage="python FAQ_main.py <command> [<args>]")
    parser.set_defaults(func=model.pretrain)
    subparsers = parser.add_subparsers(help="pretrain/finetune/eval/predict")
    pretrain_parser = subparsers.add_parser("pretrain", help="pretrain model")
    pretrain_parser.set_defaults(func=model.pretrain)
    finetune_parser = subparsers.add_parser("finetune", help="finetune model")
    finetune_parser.set_defaults(func=model.finetune)
    eval_parser = subparsers.add_parser("eval", help="evaluate model")
    eval_parser.set_defaults(func=model.evaluate_model)
    predict_parser = subparsers.add_parser("predict", help="test model predict")
    predict_parser.set_defaults(func=model.predict)

    parser.add_argument("--table_name", type=str, default="t_nlp_qa_faq")
    parser.add_argument("--sql_host", type=str, default="localhost")
    parser.add_argument("--sql_user", type=str, default="faq")
    parser.add_argument("--sql_passwd", type=str, default="123456")
    parser.add_argument("--sql_charset", type=str, default="utf8mb4")
    parser.add_argument("--sql_db", type=str, default="qa100")

    pretrain_parser.add_argument("--train_file", type=str, default="input/data/faq/pre_train_data", help="Input train file.")
    pretrain_parser.add_argument("--vocab_file",   default="input/data/faq/vocab", help="Input vocab file.")
    pretrain_parser.add_argument("--model_save_dir", type=str, default="models/FAQ/pretrain_model",  help="Specified the directory in which the model should stored.")
    pretrain_parser.add_argument("--lstm_dim", type=int, default=500, help="Dimension of LSTM cell.")
    pretrain_parser.add_argument("--embedding_dim", type=int, default=1000, help="Dimension of word embedding.")
    pretrain_parser.add_argument("--layer_num", type=int, default=1, help="LSTM layer num.")
    pretrain_parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    pretrain_parser.add_argument("--train_step", type=int, default=10000, help="Number of training steps.")
    pretrain_parser.add_argument("--warmup_step", type=int, default=5000, help="Number of warmup steps.")
    pretrain_parser.add_argument("--learning_rate", type=float, default=1e-5, help="The initial learning rate")
    pretrain_parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    pretrain_parser.add_argument("--seed", type=int, default=0, help="Random seed value.")
    pretrain_parser.add_argument("--print_step", type=int, default=1000, help="Print log every x step.")
    pretrain_parser.add_argument("--max_predictions_per_seq", type=int, default=10,
                        help="For each sequence, predict x words at most.")
    pretrain_parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay rate")
    pretrain_parser.add_argument("--clip_norm", type=float, default=1, help='Clip normalization rate.')
    pretrain_parser.add_argument("--max_seq_len", type=int, default=100, help="Max seqence length.")
    pretrain_parser.add_argument("--use_queue", type=int, default=0, help="Whether or not using a queue for input.")
    pretrain_parser.add_argument("--init_checkpoint_file", type=str, default="", help="Initial checkpoint")
    pretrain_parser.add_argument("--enqueue_thread_num", type=int, default=5, help="Enqueue thread count.")
    # pretrain_parser.add_argument("--representation_type", type=str, default="lstm", help="representation type include:lstm, transformer")
    pretrain_parser.add_argument("--representation_type", type=str, default="transformer", help="representation type include:lstm, transformer")

    #transformer args
    pretrain_parser.add_argument("--initializer_range", type=float, default="0.02", help="Embedding initialization range")
    pretrain_parser.add_argument("--max_position_embeddings", type=int, default=140, help="max position num")
    pretrain_parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    pretrain_parser.add_argument("--num_hidden_layers", type=int, default=12, help ="num hidden layer")
    pretrain_parser.add_argument("--num_attention_heads", type=int, default=12, help="num attention heads")
    pretrain_parser.add_argument("--intermediate_size", type=int, default=1024, help="intermediate_size")

    finetune_parser.add_argument("--train_file", type=str, default="input/data/faq/train_data", help="Input train file.")
    finetune_parser.add_argument("--dev_file", type=str, default="input/data/faq/dev_data", help="Input dev file.")
    finetune_parser.add_argument("--vocab_file", type=str, default="input/data/faq/vocab", help="Input vocab file.")
    finetune_parser.add_argument("--output_id2label_file", type=str, default="models/FAQ/finetune_model/id2label.has_init", help="File containing (id, class label) map.")
    finetune_parser.add_argument("--model_save_dir", type=str, default="models/FAQ/finetune_model", help="Specified the directory in which the model should stored.") 
    finetune_parser.add_argument("--lstm_dim", type=int, default=500, help="Dimension of LSTM cell.")
    finetune_parser.add_argument("--embedding_dim", type=int, default=1000, help="Dimension of word embedding.")
    finetune_parser.add_argument("--opt_type", type=str, default='adam', help="Type of optimizer.")
    finetune_parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    finetune_parser.add_argument("--epoch", type=int, default=30, help="Epoch.")
    finetune_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    finetune_parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    finetune_parser.add_argument("--seed", type=int, default=1, help="Random seed value.")
    finetune_parser.add_argument("--print_step", type=int, default=1000, help="Print log every x step.")
    finetune_parser.add_argument("--init_checkpoint_file", type=str, default='models/FAQ/pretrain_model/lm_pretrain.ckpt-10000',  help="Initial checkpoint (usually from a pre-trained model).") 
    finetune_parser.add_argument("--max_len", type=int, default=100, help="Max seqence length.")
    finetune_parser.add_argument("--layer_num", type=int, default=2, help="LSTM layer num.")

    finetune_parser.add_argument("--representation_type", type=str, default="lstm", help="representation type include:lstm, transformer")

    # transformer args
    finetune_parser.add_argument("--initializer_range", type=float, default="0.02", help="Embedding initialization range")
    finetune_parser.add_argument("--max_position_embeddings", type=int, default=140, help="max position num")
    finetune_parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    finetune_parser.add_argument("--num_hidden_layers", type=int, default=12, help="num hidden layer")
    finetune_parser.add_argument("--num_attention_heads", type=int, default=12, help="num attention heads")
    finetune_parser.add_argument("--intermediate_size", type=int, default=1024, help="intermediate_size")

    eval_parser.add_argument("--input_file", type=str, default="input/data/faq/test_data", help="Input file for prediction.")
    eval_parser.add_argument("--vocab_file", type=str, default="input/data/faq/vocab", help="Input train file.")
    eval_parser.add_argument("--model_path", type=str, default="", help="Path to model file.")
    eval_parser.add_argument("--model_dir", type=str, default="models/FAQ/finetune_model", help="Directory which contains model.")
    eval_parser.add_argument("--output_file", type=str, default="models/FAQ/result")
    eval_parser.add_argument("--id2label_file", type=str, default="models/FAQ/finetune_model/id2label.has_init", help="File containing (id, class label) map.")
    # eval_parser.add_argument("--eval_accept_threshold", type=str, default=faq_runner_config.get("admit_threshold", 0.8), help="Minimum confidence to accept FAQ answer.")

    predict_parser.add_argument("--input_sentence", type=str, default="列表页账本和事业部的字段都没有值，出库单详情中账本字段无值", help="Input sentence for prediction.")
    predict_parser.add_argument("--input_file", type=str, default="input/data/faq/ans_data", help="Input file for prediction.")
    predict_parser.add_argument("--vocab_file", type=str, default="input/data/faq/vocab", help="Input train file.")
    predict_parser.add_argument("--model_path", type=str, default="", help="Path to model file.")
    predict_parser.add_argument("--model_dir", type=str, default="models/FAQ/finetune_model", help="Directory which contains model.")
    predict_parser.add_argument("--id2label_file", type=str, default="models/FAQ/finetune_model/id2label.has_init", help="File containing (id, class label) map.")

    args = parser.parse_args()
    convert_config_paths(faq_model_config)
    command = sys.argv[1]
    if command == "pretrain":
        merge_arg_and_config(args, faq_model_config.get("pretrain", []))
    elif command == "finetune":
        merge_arg_and_config(args, faq_model_config.get("finetune", []))
    elif command == "eval":
        merge_arg_and_config(args, faq_model_config.get("eval", []))
    elif command == "predict":
        merge_arg_and_config(args, faq_model_config.get("predict", []))
    else:
        # default: pretrain
        merge_arg_and_config(args, faq_model_config.get("pretrain", []))
    args.func(args)

if __name__ == '__main__':
    main()