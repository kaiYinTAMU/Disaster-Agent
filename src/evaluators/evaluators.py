import json
import random
from typing import List, Union
import re
from collections import defaultdict
from json_repair import repair_json
import traceback

class Evaluator:
    def __init__(self):
        self.answer_marker = "The structured task plan is: "
        self.completion_count = []

    def remove_newlines(self, text):
        return re.sub(r"\n+", " ", text)
    
    def normalize_outputs(self, text: Union[str, dict, list]):
        normalized_text = self.remove_newlines(text).replace("\\", "").strip()
        return normalized_text

    def check_braces_balance(self, text):
        stack = []
        opening = {'(': ')', '{': '}', '[': ']'}
        closing = {')', '}', ']'}

        for char in text:
            if char in opening:
                stack.append(opening[char])
            elif char in closing:
                if not stack or char != stack.pop():
                    return False

        return len(stack) == 0 
    
    def isolate_answer(self, text) -> str:
        if text is None:
            return None
        assert isinstance(text, str)

        text = self.normalize_outputs(text)

        if not self.check_braces_balance(text):
            return None
        else:    
            pattern = r'\[\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}(?:\s*,\s*\{(?:[^{}]|\{[^{}]*\})*\})*)\s*\]'
            match = re.search(pattern, text, flags=re.S)
            if match:
                last_match = match.group(0)
                return last_match        
        return None

    def find_most_confident_answer(self, user_question, completions, promptbuilder, io):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list) 
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                if model_answer is None:
                    continue
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equivalence(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        if len(answer2completions.keys()) == 0:
            random_id = random.randrange(len(completions))
            random_completion = completions[random_id]
            return (random_completion, random_completion, random_id, 1e-3)
        
        if None in answer2ids:
            del answer2ids[None]
        if None in answer2completions:
            del answer2completions[None]
        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        
        most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
        sorted_answers = sorted(answer2completions.keys(), key=lambda x: len(answer2completions[x]), reverse=True)
        if len(sorted_answers) > 1 and len(answer2completions[sorted_answers[0]])  == len(answer2completions[sorted_answers[1]]):
            
            check_io_input = promptbuilder.build_check_confidence_prompt(user_question = user_question,
                                                                         output1=answer2completions[sorted_answers[0]][0],
                                                                         output2=answer2completions[sorted_answers[1]][0])
            check_output_list = io.generate(
                model_input=check_io_input, max_tokens=10, num_return=5, stop=["\n", "\n\n"]
            )
            try:
                check_output_list = [z.strip()[0] for z in check_output_list] 
                one_count = check_output_list.count('1')
            except:
                one_count = 5
            if one_count >= 3:
                most_confident_answer = sorted_answers[0]
            else:
                most_confident_answer = sorted_answers[1]

        assert (
            len(answer2completions[most_confident_answer]) > 0
        ), "There are no completions for the most confident answer."
        
        confidence = len(answer2completions[most_confident_answer]) / len(completions)
        assert confidence > 0
        return (
            most_confident_answer,
            answer2completions[most_confident_answer][0],  
            answer2ids[most_confident_answer][0],
            confidence,
        ) 
    
    def check_answers_equivalence(self, model_answer, existing_answer):
        """
        Check if two JSON-encoded answers are equivalent in structure and content.
        Uses deep structural comparison via self.deep_equal.
        """
        if not model_answer:
            return False

        try:
            model_answer = json.loads(self.remove_newlines(model_answer))
            existing_answer = json.loads(self.remove_newlines(existing_answer))
        except json.JSONDecodeError:
            print("Invalid JSON format in model or existing answer.")
            return False

        # Both must be arrays of equal length
        if not isinstance(model_answer, list) or not isinstance(existing_answer, list):
            return False
        if len(model_answer) != len(existing_answer):
            return False

        # Compare each item deeply
        for a, b in zip(model_answer, existing_answer):
            if not self.deep_equal(a, b):
                return False

        return True
    
    # def deep_equal(self, obj1, obj2):
    #     if isinstance(obj1, dict) and isinstance(obj2, dict):
    #         if set(obj1.keys()) != set(obj2.keys()):
    #             return False
    #         for k in obj1.keys():
    #             if k == "dependence_content" and obj1[k] is not None and obj2[k] is not None:
    #                 if isinstance(obj1[k], dict) and isinstance(obj2[k], dict):
    #                     if not self.deep_equal(list(obj1[k].values()), list(obj2[k].values())):
    #                         return False
    #                     else:
    #                         continue
    #                 else:
    #                     if not self.deep_equal(obj1[k], obj2[k]):
    #                         return False
    #                     else:
    #                         continue

    #             if k not in obj2.keys() or not self.deep_equal(obj1[k], obj2[k]):
    #                 return False                                                         
    #         return True

    #     elif isinstance(obj1, list) and isinstance(obj2, list):
    #         if len(obj1) != len(obj2):
    #             return False
    #         return all(self.deep_equal(a, b) for a, b in zip(sorted(obj1), sorted(obj2)))

    #     else:
    #         return obj1 == obj2

    def deep_equal(self, obj1, obj2):
        """
        Recursively check deep equality between two JSON-like Python objects.
        Handles dicts, lists, and primitive types.
        Special case: 'dependence_content' only compares its values, not keys.
        """
        # Type mismatch or None mismatch
        if type(obj1) != type(obj2):
            return False

        # ---- Dict comparison ----
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False

            for k in obj1:
                v1, v2 = obj1[k], obj2[k]

                # General recursive comparison
                if not self.deep_equal(v1, v2):
                    return False

            return True

        # ---- List comparison ----
        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(self.deep_equal(a, b) for a, b in zip(obj1, obj2))

        # ---- Base case ----
        else:
            return obj1 == obj2

    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count
        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2counts = defaultdict(list)

        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None
        
        for answer, completions in answer2completions.items():
            answer2counts[answer] = len(completions)
        
        self.completion_count.append(answer2counts)

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def extract_answer_from_model_completion(self, completion) -> str:

        assert isinstance(completion, str)

        answer_split = self.isolate_answer(completion)
        
        if answer_split is None or len(answer_split)==0:
            return None
        
        json_str = None
        try:
            loaded_json = json.loads(answer_split)
            json_str = json.dumps(loaded_json)           
        except Exception as e:
            try:
                repaired_json = json.loads(repair_json(answer_split))
                json_str = json.dumps(repaired_json)
            except Exception as e:
                print(traceback.format_exc())
        
        return json_str

    def extract_answer_from_gold_solution(self, solution) -> str:
        return json.dumps(solution)
    
    def check_tools_correctness(self, model_answer, gt_answer):
        """
        Check if the (step, agent) pairs in model and ground truth answers match exactly.
        """
        if not model_answer:
            return False

        if not isinstance(model_answer, str) or not isinstance(gt_answer, str):
            return False

        # Parse JSON safely
        try:
            model_answer = json.loads(model_answer)
            gt_answer = json.loads(gt_answer)
        except json.JSONDecodeError:
            print("Invalid JSON format in model or ground truth answer.")
            return False

        # Must be lists of equal length
        if not isinstance(model_answer, list) or not isinstance(gt_answer, list):
            return False
        if len(model_answer) != len(gt_answer):
            return False

        # Extract (step, agent) pairs
        model_tools = [
            (item.get("step"), item.get("agent"))
            for item in model_answer
            if isinstance(item, dict) and "step" in item and "agent" in item
        ]
        gt_tools = [
            (item.get("step"), item.get("agent"))
            for item in gt_answer
            if isinstance(item, dict) and "step" in item and "agent" in item
        ]

        # If the structure differs (missing fields), fail early
        if len(model_tools) != len(gt_tools):
            return False

        # Direct list comparison (order matters)
        return model_tools == gt_tools

    def check_parameters_correctness(self, model_answer, gt_answer):
        """
        Check if the 'input' parameters match for each step between model and ground truth.
        """
        if not model_answer:
            return False

        if not isinstance(model_answer, str) or not isinstance(gt_answer, str):
            return False

        print(model_answer)

        # Try parsing JSON
        try:
            model_answer = json.loads(model_answer)
            gt_answer = json.loads(gt_answer)
        except json.JSONDecodeError:
            print("Invalid JSON format in model or ground truth answer.")
            return False

        # Both must be arrays of equal length
        if not isinstance(model_answer, list) or not isinstance(gt_answer, list):
            return False
        if len(model_answer) != len(gt_answer):
            return False

        print(model_answer)

        # Compare each step one by one
        for model_item, gt_item in zip(model_answer, gt_answer):
            # Basic structural checks
            try:
                if model_item.get("step") != gt_item.get("step"):
                    return False
                if model_item.get("agent") != gt_item.get("agent"):
                    return False
            except Exception as e:
                print("Error during comparison:", e)
                print("model_answer:", model_answer)

            model_inputs = model_item.get("inputs", {})
            gt_inputs = gt_item.get("inputs", {})

            # Deep equality check for the inputs dictionary
            if not self.deep_equal(model_inputs, gt_inputs):
                return False
            
            model_outputs = model_item.get("outputs", [])
            gt_outputs = model_item.get("outputs", [])

            # Deep equality check for the outputs dictionary
            if not self.deep_equal(model_outputs, gt_outputs):
                return False

        # All steps passed
        return True

    def check_dependencies_correctness(self, model_answer, gt_answer):
        """
        Check whether 'dependence' lists and the *values* of 'dependence_content'
        match between model and ground truth answers for each step.
        """
        if not model_answer:
            return False

        if not isinstance(model_answer, str) or not isinstance(gt_answer, str):
            return False

        # Parse JSON safely
        try:
            model_answer = json.loads(model_answer)
            gt_answer = json.loads(gt_answer)
        except json.JSONDecodeError:
            print("Invalid JSON format in model or ground truth answer.")
            return False

        # Must be lists of equal length
        if not isinstance(model_answer, list) or not isinstance(gt_answer, list):
            return False
        if len(model_answer) != len(gt_answer):
            return False

        # Step-by-step comparison
        for model_item, gt_item in zip(model_answer, gt_answer):
            # Step/agent identity check
            if model_item.get("step") != gt_item.get("step"):
                return False
            if model_item.get("agent") != gt_item.get("agent"):
                return False

            # Check dependence lists
            if not self.deep_equal(sorted(model_item.get("dependence", [])), sorted(gt_item.get("dependence", []))):
                return False

            # Extract and sort dependence_content values (ignore keys)
            model_dependence_content = model_item.get("dependence_content")
            gt_dependence_content = gt_item.get("dependence_content")
            
            if model_dependence_content is None and gt_dependence_content is None:
                continue
            elif (model_dependence_content is None) != (gt_dependence_content is None):
                return False

            if not self.deep_equal(model_dependence_content, gt_dependence_content):
                return False

        # If all matched
        return True