#python test/test_generate.py --train_test_json_data train.jsonl --max_threads 1 --num_chain_of_thought 2 --use_fewshot True

import os
import gc
import time
import traceback
from dotenv import load_dotenv

from cot import ChainOfThoughts
from tot import TreeOfThoughts
from rap import RAP

from utils.common_utils import read_jsonl
from config.args import parse_args, post_process_args
from interfaces.IO_Interface import IO_Interface
from evaluators.evaluators import Evaluator

import multiprocessing
import concurrent.futures
import torch
import copy
import wandb

def worker_process(args, data_item, model_name, tokenizer_name):
    """
    Function executed in each worker process.
    Reconstructs Evaluator and IO locally.
    """

    local_args = copy.deepcopy(args)

    generation_kwargs = {
            "max_tokens": local_args.max_tokens,
            "temperature": local_args.temperature,
            "top_p": local_args.top_p
        }
        
    io = IO_Interface(local_args.api, model_name, tokenizer_name, generation_kwargs)
    evaluator = Evaluator()

    if local_args.test_type in ["cot", "sc"]:
        baseline_model = ChainOfThoughts(io, evaluator, local_args.use_fewshot, local_args.num_chain_of_thought)
    elif local_args.test_type == "tot":
        baseline_model = TreeOfThoughts(io, evaluator, local_args.num_generate_sample, local_args.num_evaluate_sample, local_args.n_select_sample)
    elif local_args.test_type == "rap":
        baseline_model = RAP(io, evaluator, local_args.n_sample_subquestion, local_args.max_depth_allowed, local_args.n_sample_confidence, local_args.w_exp, local_args.r_alpha, local_args.r1_default, local_args.num_rollouts)
    else:
        print(f"Test type {local_args.test_type} is not implemented.")

            
    task_id = data_item.get("task_id")
    task_desc = data_item.get("task_desc")
    gt_solution = data_item.get("structured_plan")
    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

    try:
        model_answer = baseline_model.generate(task_desc)
        correct = evaluator.check_answers_equivalence(model_answer, gt_answer)
        tools_correct = evaluator.check_tools_correctness(model_answer, gt_answer)
        parameters_correct = evaluator.check_parameters_correctness(model_answer, gt_answer)
        dependencies_correct = evaluator.check_dependencies_correctness(model_answer, gt_answer)

        return (model_answer, correct, tools_correct, parameters_correct, dependencies_correct)
    except Exception as e:
        traceback.format_exc()
        return None, False, False, False, False

def worker_thread(args, data_item, model_name, tokenizer_name):
    """
    Thread worker for OpenAI.
    Reuses evaluator and generator passed from the main thread.
    """
    local_args = copy.deepcopy(args)
    generation_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
    io = IO_Interface(local_args.api, model_name, tokenizer_name, generation_kwargs)
    evaluator = Evaluator()

    if local_args.test_type in ["cot", "sc"]:
        baseline_model = ChainOfThoughts(io, evaluator, local_args.use_fewshot, local_args.num_chain_of_thought)
    elif local_args.test_type == "tot":
        baseline_model = TreeOfThoughts(io, evaluator, local_args.num_generate_sample, local_args.num_evaluate_sample, local_args.n_select_sample)
    elif local_args.test_type == "rap":
        baseline_model = RAP(io, evaluator, local_args.n_sample_subquestion, local_args.max_depth_allowed, local_args.n_sample_confidence, local_args.w_exp, local_args.r_alpha, local_args.r1_default, local_args.num_rollouts)
    else:
        print(f"Test type {local_args.test_type} is not implemented.")

    task_id = data_item.get("task_id")
    task_desc = data_item.get("task_desc")
    gt_solution = data_item.get("structured_plan")
    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

    try:
        model_answer = baseline_model.generate(task_desc)

        correct = evaluator.check_answers_equivalence(model_answer, gt_answer)
        tools_correct = evaluator.check_tools_correctness(model_answer, gt_answer)
        parameters_correct = evaluator.check_parameters_correctness(model_answer, gt_answer)
        dependencies_correct = evaluator.check_dependencies_correctness(model_answer, gt_answer)

        return (model_answer, correct, tools_correct, parameters_correct, dependencies_correct)

    except Exception as e:
        traceback.format_exc()
        print(f"Error in worker_thread task {task_id}: {e}")
        return None, False, False, False, False

def main(args):
    """
    Main function to handle the execution of the script based on parsed arguments.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        None
    """

    load_dotenv()

    run = wandb.init(entity="agents-research", project="disaster-management-agent", config=args)

    assert args.api in ["openai", "huggingface", "vllm"], "Only OpenAI API, vLLM and HuggingFace models are supported."
    
    num_gpus = torch.cuda.device_count() if args.api in ["huggingface"] and torch.cuda.is_available() else 0

    if args.api in ["openai"]:
        model_name = args.model_ckpt
        tokenizer_name = None
        ExecutorClass = concurrent.futures.ThreadPoolExecutor
        num_workers = args.max_threads if args.max_threads > 0 else 1

    elif args.api in ["huggingface", "vllm"]:
        multiprocessing.set_start_method("spawn", force=True)
        model_name = args.model_ckpt
        tokenizer_name = args.tokenizer_ckpt or args.model_ckpt
        num_workers = 20
        ExecutorClass = concurrent.futures.ProcessPoolExecutor

    test_file = os.path.join(args.data_root, args.train_test_json_data)
    if args.task_num == 0:
        data_item_list = read_jsonl(test_file)
    else:
        data_item_list = read_jsonl(test_file)[args.task_num - 1 : args.task_num]
    
    total_correct = 0
    total_correct_tools = 0
    total_correct_parameters = 0
    total_correct_dependencies = 0
    num_tested = 0

    start_time = time.time()
    
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = {}
        for i, data_item in enumerate(data_item_list):
            try:
                if args.api in ["huggingface", "vllm"]:
                    futures[executor.submit(worker_process, args, data_item, model_name, tokenizer_name)] = data_item
                else:
                    futures[executor.submit(worker_thread, args, data_item, model_name, tokenizer_name)] = data_item
            except Exception as e:
                # keep original behaviour of skipping failures
                print(f"Error on item {i}: {e}")
                continue
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                model_answer, correct, tools_correct, parameters_correct, dependencies_correct = future.result()
            except Exception as e:
                print(f"Future {i} failed: {e}")
                traceback.format_exc()
                continue

            print("="*50)
            
            print(f"get result for {futures[future].get('task_id')}!\n")
            print(f"Task {futures[future].get('task_id')} Ground Truth: {futures[future].get('structured_plan')}\n")
            print(f"Task {futures[future].get('task_id')} Model Answer: {model_answer}\n")

            num_tested += 1
            total_correct += int(correct)
            total_correct_tools += int(tools_correct)
            total_correct_parameters += int(parameters_correct)
            total_correct_dependencies += int(dependencies_correct)

            print(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
            print(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
            print(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
            print(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n") 

            print("="*50)

            with open(os.path.join(args.run_outputs_dir, f"{args.test_type}_result.txt"), "w") as f:
                if not args.disable_answer_selection:
                    f.write(f"Num tested: {num_tested}\n")
                    f.write(f"Num correct: {total_correct}\n")
                    f.write(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
                    f.write(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
                    f.write(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
                    f.write(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n")   
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    average_time = elapsed_time / num_tested
    
    minutes, seconds = divmod(elapsed_time, 60)
    average_mins, average_secs = divmod(average_time, 60)

    run.log({"tools_accuracy": f"{(total_correct_tools/(num_tested))*100:.2f}\n", "parameters_accuracy": f"{(total_correct_parameters/(num_tested))*100:.2f}\n", "dependencies_accuracy": f"{(total_correct_dependencies/(num_tested))*100:.2f}\n", "overall_accuracy": f"{(total_correct/(num_tested))*100:.2f}\n", "total_execution_time": f"{minutes} min {seconds: .2f} sec", "average_execution_time": f"{average_mins} min {average_secs: .2f} sec"})

    print("Execution completed successfully.")
    print(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec")
    print(f"Average time taken: {average_mins} min {average_secs: .2f} sec")

    run.finish()
    
    with open(os.path.join(args.run_outputs_dir, f"{args.test_type}_result.txt"), "a") as f:
        f.write(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec")
        f.write(f"Average time taken: {average_mins} min {average_secs: .2f} sec")                

if __name__ == "__main__":
    args = parse_args()
    args = post_process_args(args)
    print(args)
    main(args)