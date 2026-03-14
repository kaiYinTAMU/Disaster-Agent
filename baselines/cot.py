from collections import defaultdict
import os
import random
from typing import List, Tuple
from evaluators.evaluators import Evaluator
from utils.common_utils import read_json, read_yaml
import traceback

class ChainOfThoughts:
    def __init__(self, io, evaluator, use_fewshot, num_chain_of_thought):
        self.io = io
        self.evaluator = evaluator
        self.use_fewshot = use_fewshot
        self.num_chain_of_thought = num_chain_of_thought
        self.prompt = read_yaml("baselines/baseline_prompts/cot_prompt.yaml")["CHAIN_OF_THOUGHT"]
        self.tools_desc = read_json("src/interfaces/tools/tools_manifest.json")

    def most_likely_answer(self, user_question, io_output_list):
        if len(io_output_list) == 1:
            most_confident_answer = self.evaluator.extract_answer_from_model_completion(io_output_list[0])
            confidence = 1
        else:  
            answer2completions = defaultdict(list) 
            for id, c in enumerate(io_output_list):
                try:
                    model_answer = self.evaluator.extract_answer_from_model_completion(c)
                    if model_answer is None:
                        continue
                    has_existed = False
                    for existing_answer in answer2completions.keys():
                        if self.evaluator.check_answers_equivalence(model_answer, existing_answer):
                            assert not has_existed
                            has_existed = True
                            answer2completions[existing_answer].append(c)
                            break
                    if not has_existed:
                        answer2completions[model_answer].append(c)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    continue

            if len(answer2completions.keys()) == 0:
                random_id = random.randrange(len(io_output_list))
                random_completion = io_output_list[random_id]
                return (random_completion, random_completion, random_id, 1e-3)
            
            if None in answer2completions:
                del answer2completions[None]
            assert len(answer2completions.keys()) > 0, "There are no valid completions."

            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))

        return most_confident_answer

    def generate(self, user_question):
        if self.use_fewshot:
            examples_str = "Examples: "
            for example in self.prompt["examples"]:
                examples_str += "\n" + f"{example['input']}"
                examples_str += "\n" + f"{example['output']}"
        else:
            examples_str = ""

        system_prompt = self.prompt["system_prompt"].format(agents_desc = self.tools_desc, examples = examples_str) #.replace("{{", "{").replace("}}", "}")
        user_prompt = self.prompt["user_prompt"].format(task_desc = user_question) #.replace("{{", "{").replace("}}", "}")

        io_input = system_prompt + "\n" + user_prompt

        io_output_list = self.io.generate(io_input, num_return=self.num_chain_of_thought)
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list] 

        return self.most_likely_answer(user_question, cleaned_io_output_list)