import traceback

from mcts.mcts_search import search_for_answers
from evaluators.evaluators import Evaluator
from utils.logger import setup_main_logger
from distill import main as distill_main
from utils.common_utils import read_json

def main(args, user_question):
    try:
        model_answer = None

        main_logger, log_queue, log_listener = setup_main_logger(args.logdir)

        if args.api in ["openai", "huggingface", "vllm"]:
            model_name = args.model_ckpt
            tokenizer_name = args.tokenizer_ckpt or args.model_ckpt

        distill_main(args, user_question)

        if args.if_use_cards:
            args.reason_structure = read_json(args.reuse_dir)

        evaluator = Evaluator()

        model_answer, model_completion = search_for_answers(args, evaluator, None, user_question, 0, None, model_name, tokenizer_name, log_queue)

        return model_answer, model_completion
    except Exception as e:
        traceback.format_exc()
