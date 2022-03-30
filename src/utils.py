import argparse

def common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_prompt", action="store_true", help="Whether to use prompt.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size train.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="No. steps before backward pass.",)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs", )
    parser.add_argument("--max_steps", default=-1, type=int, help="If>0: no. train steps. Overrides num_train_epochs.",)
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=10, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization") #1234
    parser.add_argument("--patience", type=int, default=5, help="Stop training if eval score does not imporve for patience time")

    parser.add_argument(
        "--model_type",
        default='roberta',
        type=str,
        help="Model type selected in the list: [roberta, bert, roberta_prompt]",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="allenai/biomed_roberta_base",
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )

    parser.add_argument(
        "--params_output_dir",
        default='./params/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_output_dir",
        default='./eval_output/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_input_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )

    parser.add_argument(
        "--save_model_name_as",
        default=None,
        type=str,
        help="Name the trained model",
    )

    return parser