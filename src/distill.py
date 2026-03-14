import os
import re
import sys

from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
sys.path.append(".")

import math
from tqdm import tqdm
import json
from itertools import groupby

from interfaces.IO_Interface import IO_Interface
from utils.common_utils import read_jsonl, read_txt, save_json, read_json
from evaluators.evaluators import Evaluator
import ast
from config.args import parse_args

class PathExtractor:
    def __init__(self, k, root_dir, file_dir, output_dir):
        self.root_dir = root_dir
        self.file_dir = file_dir
        self.evaluator = Evaluator()
        answer_sheets_dir = os.path.join(self.file_dir, "answer_sheets")
        pattern = re.compile(r"Task (\d+) - Answer.json")
        self.num = max(
                        [int(pattern.match(f).group(1)) for f in os.listdir(answer_sheets_dir) if pattern.match(f)],
                        default=None
                    )
        self.k = k
        self.output_dir = output_dir
        self.train_path_solutions_dir = os.path.join(self.output_dir, "train_path_solutions.json")
        self.train_path_questions_dir = os.path.join(self.output_dir, "train_path_questions.json")
        
        self.store = {}
        self.path_question = {}
    
    def extract_path(self):
        self.correct = 0
        self.total_correct = 0
        for i in range(1, self.num + 1):
            self.total_correct += 1
            answer_file = os.path.join(self.file_dir, f"answer_sheets/Task {i} - Answer.json")
            solution_file = os.path.join(self.file_dir, f"answer_sheets/Task {i} - Final Solutions.json")
            
            with open(answer_file, "r") as f:
                answers = json.load(f)

            try:
                with open(solution_file, "r") as f:
                    solutions = json.load(f)
                    sorted_solutions = self.sort_solutions(solutions) # metrics
                    # selected_solution = self.find_valid_solution(sorted_solutions, answers)
                    selected_solutions = self.find_valid_k_solutions(sorted_solutions, answers, k=5)
                    
                    if not selected_solutions:
                            print(f"Skipping Task {i} as no valid solution found.")
                            continue
                    
                    for selected_solution in selected_solutions:
                        select_trace = selected_solution["trace"]["0"]
                        confidence_flag, leaf_confidence = self.get_leaf_confidence(selected_solution)
                        path = tuple([x[0] for x in select_trace["path"]])

                        if path not in self.store:
                            self.path_question[path] = {answers["problem"]}
                            self.store[path] = [{
                                "id": 0,
                                "question": answers["problem"],
                                "gold_solution": answers["gold_solution"],
                                "gold_answer": answers["gold_answer"],
                                "model_solution": "Question: "+select_trace['answers'][0][1]+"\nAnswer:\n" + "\n".join([f"Step {j+1}: ({path[j+1].lower()}) "+x[1] for j, x in enumerate(select_trace['answers'][1:])]),
                                "model_answer": selected_solution["model_answer"],
                                "leaf_confidence": leaf_confidence
                            }]
                        else:
                            self.path_question[path].add(answers["problem"])
                            path_exist_len = len(self.store[path])
                            self.store[path].append({
                                "id": path_exist_len,
                                "question": answers["problem"],
                                "gold_solution": answers["gold_solution"],
                                "gold_answer": answers["gold_answer"],
                                "model_solution": "Question: "+select_trace['answers'][0][1]+"\nAnswer:\n" + "\n".join([f"Step {j+1}: ({path[j+1].lower()}) "+x[1] for j, x in enumerate(select_trace['answers'][1:])]),
                                "model_answer": selected_solution["model_answer"],
                                "leaf_confidence": leaf_confidence
                            })
            
            except:
                continue
        
        self.store_str_keys = {str(key): value for key, value in self.store.items()}
        self.path_question_str_keys = {str(key): list(value) for key, value in self.path_question.items()}
        if not os.path.exists(self.train_path_solutions_dir) and not os.path.exists(self.train_path_questions_dir):
            self.save_results()

    def sort_solutions(self, solutions):
        for x in solutions:
            if x["rollout_id"] is None:
                x["rollout_id"] = 0
        sorted_solutions = sorted(solutions, key=lambda x: x["rollout_id"], reverse=True)
        grouped_solutions = [
            list(group) for _, group in groupby(
                sorted_solutions,
                key=lambda x: x["rollout_id"]
            )
        ]

        final_sorted_solutions = []
        for group in grouped_solutions:
            if len(group) > 1:
                for x in group:
                    try:
                        x['value'] = x["trace"]["0"]["chain_of_thought"]["value"]
                    except:
                        len_keys = len(list(x["trace"].keys()))
                        try:
                            x['value'] = x["trace"][f'{len_keys-1}']["chain_of_thought"]["value"]
                        except:
                            try:
                                x['value'] = x["trace"][f'{len_keys-1}']['subanswer']['value']
                            except:
                                x['value'] = 0
                try:
                    group = sorted(
                        group,
                        key=lambda x: (
                            len(x["trace"]["0"]["path"])*self.k 
                            -x['value']*(1-self.k)
                        )
                    )
                except:
                    group = sorted(
                        group,
                        key=lambda x: (
                            len(x["trace"]["0"]["path"])
                        )
                    )
            final_sorted_solutions.extend(group)

        return final_sorted_solutions

    def find_valid_solution(self, sorted_solutions, answers):
        ans = self.evaluator.extract_answer_from_model_completion(sorted_solutions[0]["trace"]["0"]['answers'][-1][-1])
        self.correct += int(self.evaluator.check_answers_equivalence(ans, answers["gold_answer"]))

        for solution in sorted_solutions:
            select_trace = solution["trace"]["0"]
            model_answer = self.evaluator.extract_answer_from_model_completion(select_trace['answers'][-1][-1])
            if self.evaluator.check_answers_equivalence(model_answer, answers["gold_answer"]):
                solution["model_answer"] = model_answer
                return solution
        return None

    def find_valid_k_solutions(self, sorted_solutions, answers, k):
        ans = self.evaluator.extract_answer_from_model_completion(sorted_solutions[0]["trace"]["0"]['answers'][-1][-1])
        self.correct += int(self.evaluator.check_answers_equivalence(ans, answers["gold_answer"]))

        valid_solutions = []
        for solution in sorted_solutions:
            select_trace = solution["trace"]["0"]
            model_answer = self.evaluator.extract_answer_from_model_completion(select_trace['answers'][-1][-1])
            if self.evaluator.check_answers_equivalence(model_answer, answers["gold_answer"]):
                solution["model_answer"] = model_answer
                valid_solutions.append(solution)
        
        if len(valid_solutions) == 0:
            return None
        else:
            return valid_solutions[:k]
    
    def get_leaf_confidence(self, solution):
        confidence_flag = 0
        temp_trace = solution["trace"]
        select_trace = temp_trace["0"]
        try:
            leaf_confidence = select_trace["chain_of_thought"]["value"]
            confidence_flag = 1
        except:
            for j in range(1, len(list(temp_trace.keys()))):
                try:
                    leaf_confidence = temp_trace[str(j)]["chain_of_thought"]["value"]
                    confidence_flag = 1
                    break
                except:
                    continue
        if confidence_flag == 0:
            leaf_confidence = 0
        return confidence_flag, leaf_confidence
    
    def save_results(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.train_path_solutions_dir, "w") as f:
            json.dump(self.store_str_keys, f)
        with open(self.train_path_questions_dir, "w") as f:
            json.dump(self.path_question_str_keys, f)


class RephrasingHandler:
    def __init__(self, rephrasing_prompt_template: str, io: IO_Interface):
        self.rephrasing_prompt_template = rephrasing_prompt_template
        self.io = io
    
    def generate_rephrased_question(self, questions: list):
        io_inputs = []
        for question in questions:
            io_input = self.rephrasing_prompt_template
            io_input += "\n\n"
            io_input += "Original Question: " + question + "\n"
            io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
            io_inputs.append(io_input)
        
        io_outputs = self.io.generate(model_input=io_inputs, max_tokens=512, num_return=1, stop=["\n", "\n\n"])
        results = []
        for io_output in io_outputs:

            count_conditions = io_output.count("Condition") + 1
            results.append((io_output, count_conditions))
        
        return results, count_conditions


class ProblemDecomposer:
    def __init__(self, io: IO_Interface, rephrasing_prompt_template: str, structure_dir: str):
        self.io = io
        self.rephrasing_handler = RephrasingHandler(rephrasing_prompt_template, self.io)
        self.structure_dir = structure_dir

    def decompose_problems(self, train_path_questions, type):
        train_decompose_list = {}
        train_difficulty_list = {}
        
        for p, q_list in train_path_questions.items():
            
            io_outputs, _ = self.rephrasing_handler.generate_rephrased_question(q_list)
            
            for i, (io_output, count_conditions) in enumerate(io_outputs):
                if i == 0:
                    train_decompose_list[p] = [{
                        "question": q_list[i],
                        "count_conditions": count_conditions,
                        "prompt": "Condition 1: " + io_output
                    }]
                    train_difficulty_list[p] = [count_conditions]
                else:
                    train_decompose_list[p].append({
                        "question": q_list[i],
                        "count_conditions": count_conditions,
                        "prompt": "Condition 1: " + io_output
                    })
                    train_difficulty_list[p].append(count_conditions)

        return train_decompose_list, train_difficulty_list


class ProblemDecompositionManager:  
    def __init__(self, k, io, structure_dir, root_dir, file_dir, output_dir, attribute_type, similarity_type):
        self.root_dir = root_dir
        self.structure = structure_dir
        self.structure_dir = output_dir
        self.k = k
        self.file_dir = file_dir
        self.attribute_type = attribute_type
        self.similarity_type = similarity_type
        self.train_decompose_dir = os.path.join(self.structure_dir, f"{attribute_type}_train_decompose.json")
        self.train_path_difficulty_list_dir = os.path.join(self.structure_dir, f"{attribute_type}_train_path_difficulty_list.json")
        self.train_path_difficulty_count_dir = os.path.join(self.structure_dir, f"{attribute_type}_train_path_difficulty_count.json")
        model_name = root_dir.split("/")[1].split("/")[0].strip()
        self.store_path_difficulty_dir = os.path.join(self.structure_dir, f"{attribute_type}_|_{model_name}_path_difficulty_count.json")
        self.test_set_path_dir = "data/test.jsonl"
        
        self.load_data()
        
        self.io = io
        self.extractor = PathExtractor(self.k, root_dir, file_dir, output_dir)
        self.extractor.extract_path()
        self.train_path_solutions = self.extractor.store_str_keys
        self.train_path_questions = self.extractor.path_question_str_keys
    
        self.decomposer = ProblemDecomposer(self.io, self.rephrasing_prompt_template, self.structure_dir)

    def load_data(self):
        self.rephrasing_prompt_template = read_txt(os.path.join("src", "prompts", "distill_prompts.txt"))
        
        if not os.path.exists(self.structure_dir):
            os.makedirs(self.structure_dir)

        self.testset = read_jsonl(self.test_set_path_dir)
        
    def decompose(self):
        self.train_decompose_dict, self.train_difficulty_list_dict = self.decomposer.decompose_problems(self.train_path_questions, self.attribute_type)
        sorted_train_difficulty_list = dict(
            sorted(self.train_difficulty_list_dict.items(), key=lambda x: sum(x[1]) / len(x[1]))
        )
        
        self.train_path_difficulty_count_dict = {}
        for k, v in sorted_train_difficulty_list.items():
            self.train_path_difficulty_count_dict[k] = sum(v) / len(v)
            print(f"{k}: {sum(v) / len(v):.2f}")
        
        self.save_results()
        
    def save_results(self):
        if not os.path.exists(self.train_decompose_dir):
            save_json(self.train_decompose_dict, self.train_decompose_dir)
        if not os.path.exists(self.train_path_difficulty_list_dir):
            save_json(self.train_difficulty_list_dict, self.train_path_difficulty_list_dir)
        if not os.path.exists(self.train_path_difficulty_count_dir):
            save_json(self.train_path_difficulty_count_dict, self.train_path_difficulty_count_dir)
        if not os.path.exists(self.store_path_difficulty_dir):
            save_json(self.train_path_difficulty_count_dict, self.store_path_difficulty_dir)
    
    def find_nearest_train_path(self, count_conditions, k):
        # Find k paths with the minimum distance to count_conditions
        distances = []
        for key_str in self.train_path_difficulty_count.keys():
            key_tuple = ast.literal_eval(key_str)  
            distance = abs(len(key_tuple) - count_conditions)
            distances.append((distance, key_str))

        # Sort distances and take the nearest k
        distances.sort(key=lambda x: x[0])
        nearest_keys_strs = [key_str for _, key_str in distances[:k]]

        # Convert the nearest keys (string representation) to lists
        nearest_keys_lists = [list(ast.literal_eval(key_str)) for key_str in nearest_keys_strs]
        return nearest_keys_lists    
    
    def obtain_path(self):
        top_k = 10
        batch_size = 32
        test_question_paths = {}
        test_set_name = self.test_set_path_dir.split("/")[-2]
        if self.attribute_type == "condition":
            self.store_test_path_difficulty_dir = os.path.join(self.structure_dir, f"{test_set_name}_|_{self.attribute_type}_|_test_question_path.json")
            rephrasing_handler = self.decomposer.rephrasing_handler
            user_question_list = [item["task_desc"] for item in self.testset]
            
            if self.attribute_type == "condition":
                self.train_path_difficulty_count = read_json(self.store_path_difficulty_dir)
                io_outputs = []
                num_batches = math.ceil(len(user_question_list) / batch_size)
                for i in tqdm(range(num_batches), desc="Rephrasing batches"):
                    batch = user_question_list[i * batch_size : (i + 1) * batch_size]
                    batch_outputs, _ = rephrasing_handler.generate_rephrased_question(batch)
                    io_outputs.extend(batch_outputs)

            for i, (io_output, count_condition) in enumerate(io_outputs):
                path = self.find_nearest_train_path(count_condition, top_k)
                test_question_paths[user_question_list[i]] = path
        
        elif self.attribute_type == "semantic":
            self.store_test_path_difficulty_dir = os.path.join(self.structure_dir, f"{test_set_name}_|_{self.attribute_type}_{self.similarity_type}_|_test_question_path.json")
            model = SentenceTransformer('intfloat/e5-small-v2')

            for item in self.testset:
                test_question = item['task_desc']
                similarities = {}
                for path, questions in self.train_path_questions.items():
                    path_similarities = []
                    for question in questions:
                        sentences = ['query: ' + test_question, 'passage: ' + question]
                        embeddings = model.encode(sentences, normalize_embeddings=True)
                        similarity = embeddings[0] @ embeddings[1].T
                        path_similarities.append(similarity)     
                    
                    if self.similarity_type == "average":
                        avg_similarity = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                        similarities[path] = avg_similarity
                    elif self.similarity_type == "max":
                        max_similarity = max(path_similarities) if path_similarities else 0
                        similarities[path] = max_similarity

                top_k_paths = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
                for k, v in top_k_paths:
                    print(f"Path: {k}, Similarity: {v}")
                test_question_paths[test_question] = [list(ast.literal_eval(path)) for path, _ in top_k_paths]   
                print(len(list(test_question_paths.keys())))

        elif self.attribute_type == "rerank":
            self.store_test_path_difficulty_dir = os.path.join(self.structure_dir, f"{test_set_name}_|_{self.attribute_type}_{self.similarity_type}_|_test_question_path.json")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
            
            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")
            max_length = 8192
            task = 'Given a user query, retrieve relevant document that is the most similar'

            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

            def format_instruction(instruction, query, doc):
                if instruction is None:
                    instruction = 'Given a user query, retrieve relevant document that is the most similar'
                output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
                return output
            
            def process_inputs(pairs):
                inputs = tokenizer(
                    pairs, padding=False, truncation='longest_first',
                    return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
                )
                for i, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
                inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
                for key in inputs:
                    inputs[key] = inputs[key].to(model.device)
                return inputs
            
            @torch.no_grad()
            def compute_logits(inputs, **kwargs):
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                return scores
            
            for item in self.testset:
                print(f"Obtaining path for test id {item['task_id']} started.\n")
                test_question = item['task_desc']
                print(f"Test Question: {test_question}\n")
                similarities = {}

                for path, questions in self.train_path_questions.items(): 
                    path_similarities = []
                    question_pairs = [format_instruction(task, test_question, question) for question in questions]
                    inputs = process_inputs(question_pairs)
                    path_similarities = compute_logits(inputs)
                    
                    if self.similarity_type == "average":
                        avg_similarity = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                        print(avg_similarity)
                        similarities[path] = avg_similarity
                    elif self.similarity_type == "max":
                        max_similarity = max(path_similarities) if path_similarities else 0
                        print(max_similarity)
                        similarities[path] = max_similarity

                top_k_paths = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
                for k, v in top_k_paths:
                    print(f"Path: {k}, Similarity: {v}")
                test_question_paths[test_question] = [list(ast.literal_eval(path)) for path, _ in top_k_paths]   
                print(len(list(test_question_paths.keys())))

        save_json(test_question_paths, self.store_test_path_difficulty_dir)
        return test_question_paths   
    
    def obtain_single_path(self, user_question):
        """
        Generate the top-k train paths most similar to a single user question.
        """

        top_k = 10
        batch_size = 32
        test_question_paths = {}

        # Extract test set name for output path
        test_set_name = self.test_set_path_dir.split("/")[-2]

        # ===== CASE 1: CONDITION ATTRIBUTE =====
        if self.attribute_type == "condition":
            self.store_test_path_difficulty_dir = os.path.join(
                self.structure_dir, f"{test_set_name}_|_{self.attribute_type}_|_test_question_path.json"
            )

            rephrasing_handler = self.decomposer.rephrasing_handler
            self.train_path_difficulty_count = read_json(self.store_path_difficulty_dir)

            # Generate rephrased versions of the single user question
            io_outputs, _ = rephrasing_handler.generate_rephrased_question([user_question])

            # For each rephrased variant, find the nearest path
            for io_output, count_condition in io_outputs:
                path = self.find_nearest_train_path(count_condition, top_k)
                test_question_paths[user_question] = path

        # ===== CASE 2: SEMANTIC ATTRIBUTE =====
        elif self.attribute_type == "semantic":
            self.store_test_path_difficulty_dir = os.path.join(
                self.structure_dir,
                f"{test_set_name}_|_{self.attribute_type}_{self.similarity_type}_|_test_question_path.json"
            )

            from sentence_transformers import SentenceTransformer
            import ast

            model = SentenceTransformer('intfloat/e5-small-v2')
            similarities = {}

            # Compare user_question against all train path questions
            for path, questions in self.train_path_questions.items():
                path_similarities = []
                for question in questions:
                    sentences = [f'query: {user_question}', f'passage: {question}']
                    embeddings = model.encode(sentences, normalize_embeddings=True)
                    similarity = embeddings[0] @ embeddings[1].T
                    path_similarities.append(similarity)

                if self.similarity_type == "average":
                    similarities[path] = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                elif self.similarity_type == "max":
                    similarities[path] = max(path_similarities) if path_similarities else 0

            top_k_paths = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for k, v in top_k_paths:
                print(f"Path: {k}, Similarity: {v}")

            test_question_paths[user_question] = [list(ast.literal_eval(path)) for path, _ in top_k_paths]

        # ===== CASE 3: RERANK ATTRIBUTE =====
        elif self.attribute_type == "rerank":
            import torch
            import ast
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.store_test_path_difficulty_dir = os.path.join(
                self.structure_dir,
                f"{test_set_name}_|_{self.attribute_type}_{self.similarity_type}_|_test_question_path.json"
            )

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()

            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")
            max_length = 8192
            task = 'Given a user query, retrieve relevant document that is the most similar'

            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

            def format_instruction(instruction, query, doc):
                if instruction is None:
                    instruction = task
                return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

            def process_inputs(pairs):
                inputs = tokenizer(
                    pairs, padding=False, truncation='longest_first',
                    return_attention_mask=False,
                    max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
                )
                for i, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
                inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
                for key in inputs:
                    inputs[key] = inputs[key].to(model.device)
                return inputs

            @torch.no_grad()
            def compute_logits(inputs):
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                return scores

            similarities = {}

            # Compare user question with all training paths
            for path, questions in self.train_path_questions.items():
                question_pairs = [format_instruction(task, user_question, question) for question in questions]
                inputs = process_inputs(question_pairs)
                path_similarities = compute_logits(inputs)

                if self.similarity_type == "average":
                    similarities[path] = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                elif self.similarity_type == "max":
                    similarities[path] = max(path_similarities) if path_similarities else 0

            top_k_paths = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for k, v in top_k_paths:
                print(f"Path: {k}, Similarity: {v}")

            test_question_paths[user_question] = [list(ast.literal_eval(path)) for path, _ in top_k_paths]

        # ===== SAVE & RETURN =====
        save_json(test_question_paths, self.store_test_path_difficulty_dir)
        return test_question_paths

def main(args, user_question = None):
    generation_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            } 
    if args.api == "huggingface":
        generation_kwargs["top_k"] = args.top_k

    io = IO_Interface(api=args.api, model_name=args.model_ckpt, tokenizer_name=args.tokenizer_ckpt or args.model_ckpt, generation_kwargs=generation_kwargs)
    
    manager = ProblemDecompositionManager(  args.k, 
                                            io,
                                            args.structure_dir, 
                                            args.root_dir, 
                                            args.file_dir, 
                                            args.distill_outputs_dir,  
                                            args.attribute_type,
                                            args.similarity_type
                                        )  
    if args.attribute_type in ["condition"]:
        manager.decompose()

    if args.mode == "distill":
        test_question_paths = manager.obtain_path()
    elif args.mode == "eval":
        test_question_paths = manager.obtain_single_path(user_question)
    else:
        print("Incorrect mode")
    print(test_question_paths)
