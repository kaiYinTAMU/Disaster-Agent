import os
import ast
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import deepcopy
import threading

from mcts.mcts_generator import MCTS_Generator
from mcts.mcts_base import MCTS_Searcher
from mcts.mcts_node import Agent_MCTS_Node
from utils.agent_utils import Node_Type, stochastic_find_best_solution
from utils.common_utils import save_json
from utils.logger import setup_main_logger, get_worker_logger

def generator_thread_or_process(
    j,
    train_path,
    args_local,
    user_question,
    task_id,
    gt_answer,
    generator_info,
    log_queue
):
    """
    Worker function for a single MCTS rollout path.
    Supports both multiprocessing (HuggingFace/vLLM) and multithreading (OpenAI API).
    """

    if isinstance(threading.current_thread(), threading._MainThread):
        worker_id = f"proc_{os.getpid()}"
    else:
        worker_id = f"thread_{threading.get_ident()}"
    logger = get_worker_logger(log_queue, f"worker_{worker_id}")

    # Recreate generator locally if multiprocessing
    if isinstance(generator_info, dict):
        generator = MCTS_Generator(
            args=args_local,
            model_name=generator_info["model_name"],
            tokenizer_name=generator_info["tokenizer_name"]
        )
    else:
        generator = generator_info  # Thread-safe shared instance

    model_solutions, model_all_solutions, model_rollout_nodes, model_best_path, model_all_solution_nodes = (
        [],
        [],
        [],
        [],
        [],
    )

    try:
        if args_local.if_use_cards:
            args_local.max_depth_allowed = len(train_path)
            args_local.num_rollouts = args_local.reuse_rollouts

        # Initialize MCTS searcher
        mcts_searcher = MCTS_Searcher(
            exploration_weight=args_local.mcts_exploration_weight,
            weight_scheduler=args_local.mcts_weight_scheduler,
            num_rollouts=args_local.num_rollouts,
            discount=args_local.mcts_discount_factor,
            verbose=args_local.verbose,
        )

        # Initialize root node
        root_node = Agent_MCTS_Node(
            parent=None,
            depth=0,
            node_type=Node_Type.USER_QUESTION,
            verbose=args_local.verbose,
            generator=generator,
            user_question=user_question,
            max_depth_allowed=args_local.max_depth_allowed,
            disable_rephrase=args_local.disable_rephrase,
            disable_direct_answer=args_local.disable_direct_answer,
            disable_chain_of_thought=args_local.disable_chain_of_thought,
            disable_divide_and_conquer=args_local.disable_divide_and_conquer,
            disable_self_refine=args_local.disable_self_refine,
            if_use_cards=args_local.if_use_cards,
            reasoning_path=train_path,
        )

        # Perform rollouts
        for i in range(args_local.num_rollouts):
            try:
                logger.info(
                    f"[Task {task_id}] Rollout {i} (path {j}) started."
                )
                rollout_node = mcts_searcher.do_rollout(root_node, i)
                model_rollout_nodes.append(rollout_node)

                # Select best answer among rollouts
                if not args_local.disable_answer_selection:
                    _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
                        root_node,
                        generator.evaluator,
                        enable_potential_score=args_local.enable_potential_score,
                    )
                    if best_solution is None:
                        continue
                    model_solutions.append(best_solution)
                    model_all_solutions.append(all_solutions)
                    if (
                        chosen_node.solution_trace
                        and "path" in chosen_node.solution_trace[0]
                    ):
                        model_best_path.append(chosen_node.solution_trace[0]["path"])
                    model_all_solution_nodes.extend(all_solution_nodes)

                logger.info(
                    f"[Task {task_id}] Rollout {i} (path {j}) completed."
                )
                print(f"{i} completed")
            except Exception as e:
                logger.error(f"Error in rollout {i} (path {j}): {e}")
                traceback.print_exc()

    except Exception as e:
        logger.error(f"Exception in train_path {j}: {e}")
        traceback.print_exc()
    finally:
        # clean memory
        if hasattr(generator, "close"):
            generator.close()

    if args_local.mode != 'eval' and model_solutions:
        # Write solution JSONs safely
        os.makedirs(args_local.answer_sheets_dir, exist_ok=True)

        save_json(
            [{"rollout_id": n.rollout_id, "trace": n.solution_trace} for n in all_solution_nodes],
            os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Final Solutions.json"),
        )
        save_json(            
            [{"rollout_id": i, "trace": n.solution_trace} for i, n in enumerate(model_rollout_nodes)],
            os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Rollout Solutions.json"),
        )
        save_json(
            [
                {"rollout_id": i, "path": n.solution_trace[0].get("path", None)}
                for i, n in enumerate(model_rollout_nodes)
                if n.solution_trace
            ],
            os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Rollout Path.json"),
        )

    # return (j, model_solutions, model_all_solutions, model_rollout_nodes, model_best_path, [model_all_solution_nodes])
    return (j, model_solutions, model_all_solutions, model_best_path)

def search_for_answers(
    args, 
    evaluator, 
    original_js, 
    user_question: str, 
    task_id: int, 
    gt_answer: str, 
    model_name: str, 
    tokenizer_name:str, 
    log_queue: object
):
    """
    Perform MCTS-based answer search with parallel rollouts.
    Uses threads for OpenAI API and processes for local models (HuggingFace/vLLM).
    """
    logger = get_worker_logger(log_queue, f"search_task_{task_id}")
    logger.info(f"Starting search for task {task_id}: {user_question}")

    # Prepare train paths
    if args.if_use_cards:
        try:
            train_paths = args.reason_structure[user_question][: args.num_cards]
            train_paths = [
                list(ast.literal_eval(x)) if not isinstance(x, list) else x
                for x in train_paths
            ]
        except KeyError as e:
            logger.error("Key not found in distilled paths")
    else:
        train_paths = [None]

    print(train_paths)

    # Select executor type
    if args.api in ["huggingface", "vllm"]:
        Executor = ProcessPoolExecutor
        generator_info = {
            "model_name": args.model_ckpt,
            "tokenizer_name": args.tokenizer_ckpt or args.model_ckpt
        }
        logger.info("Using multiprocessing executor for local inference.")
    else:
        Executor = ThreadPoolExecutor
        generator_info = MCTS_Generator(args, args.model_ckpt, None)
        logger.info("Using multithreading executor for API-based inference.")

    # Run rollouts in parallel
    path_threads = 2
    results = []

    with Executor(max_workers=path_threads) as executor:
        futures = [
            executor.submit(
                generator_thread_or_process,
                j,
                train_path,
                deepcopy(args),
                user_question,
                task_id,
                gt_answer,
                generator_info,
                log_queue
            )
            for j, train_path in enumerate(train_paths)
        ]

        try:
            for future in as_completed(futures):
                results.append(future.result())
        except Exception as e:
            logger.error(f"Worker crashed: {e}")
            traceback.print_exc()

    # Sort and aggregate results
    results.sort(key=lambda x: x[0])
    model_solutions, model_all_solutions, model_best_path = (
        [],
        [],
        [],
    )

    for _, sols, all_sols, paths in results:
        model_solutions.extend(sols)
        model_all_solutions.extend(all_sols)
        model_best_path.extend(paths)

    logger.info(
        f"Collected {len(model_solutions)} solutions."
    )

    # Default evaluation results
    correct = tools_correct = parameters_correct = dependencies_correct = None

    # Evaluate answers
    if not args.disable_answer_selection and model_solutions:
        for rollout_id, (model_path, model_solution, model_all_solution) in enumerate(
            zip(model_best_path, model_solutions, model_all_solutions)
        ):
            model_answer = evaluator.extract_answer_from_model_completion(model_solution)
            model_all_answers = [
                evaluator.extract_answer_from_model_completion(a)
                for a in model_all_solution
                if a is not None
            ]

            if args.mode != 'eval' and model_answer:
                correct = evaluator.check_answers_equivalence(model_answer, gt_answer)
                correct_limit = any(
                    evaluator.check_answers_equivalence(a, gt_answer)
                    for a in model_all_answers
                    if a is not None
                )
                tools_correct = evaluator.check_tools_correctness(model_answer, gt_answer)
                parameters_correct = evaluator.check_parameters_correctness(
                    model_answer, gt_answer
                )
                dependencies_correct = evaluator.check_dependencies_correctness(
                    model_answer, gt_answer
                )

                original_js["all_model_completions"][f"rollout_{rollout_id}"] = {
                    "model_solution": model_solution,
                    "model_answer": model_answer,
                    "model_path": model_path,
                    "correct": correct,
                    "correct_limit": correct_limit,
                    "tools_correct": tools_correct,
                    "parameters_correct": parameters_correct,
                    "dependencies_correct": dependencies_correct,
                }

        best_answer, best_solution, _, _ = evaluator.stochastic_find_most_confident_answer(model_solutions)

        if args.mode != 'eval':
            correct = evaluator.check_answers_equivalence(best_answer, gt_answer)
            tools_correct = evaluator.check_tools_correctness(best_answer, gt_answer)
            parameters_correct = evaluator.check_parameters_correctness(best_answer, gt_answer)
            dependencies_correct = evaluator.check_dependencies_correctness(best_answer, gt_answer)

            logger.info(f"GT {task_id}: {gt_answer}")
            logger.info(f"Best {task_id}: {best_answer}")

            logger.info(
                f"Final evaluation for task {task_id}: Correct={correct}, Tools={tools_correct}, "
                f"Params={parameters_correct}, Deps={dependencies_correct}"
            )

            original_js["model_completion"] = best_solution
            original_js["model_answer"] = best_answer
            original_js["model_all_answer"] = model_all_solutions

            return (
                original_js,
                model_solutions,
                model_all_solutions,
                model_best_path,
                correct,
                tools_correct,
                parameters_correct,
                dependencies_correct,
            )
        else:
            return best_answer, best_solution
