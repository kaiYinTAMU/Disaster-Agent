import itertools
import json
import re
from evaluators.evaluators import Evaluator
from interfaces.IO_Interface import IO_Interface
from utils.common_utils import read_yaml, read_json

class TreeOfThoughts:
    def __init__(self, io, evaluator, num_generate_sample, num_evaluate_sample, n_select_sample):
        self.io = io
        self.evaluator = evaluator
        self.num_steps = 2
        self.num_generate_sample = num_generate_sample
        self.num_evaluate_sample = num_evaluate_sample
        self.n_select_sample = n_select_sample
        self.prompt = read_yaml("baselines/baseline_prompts/tot_prompts.yaml")["TREE_OF_THOUGHTS"] 
        self.agents_desc = read_json("src/interfaces/tools/tools_manifest.json")

    def generate_thoughts(self, x, y, step):
        """Generate multiple potential next steps (thoughts)."""
        prompt = self.prompt["generate_prompt"].format(agents_desc = self.agents_desc, user_question = x).rstrip() + y.strip()
        if step == 1:
            prompt += "\n\nThe structured task plan is: "
            # prompt += "\n\n"
        stop_tokens = [["The structured task plan is:", "Response:", "Instruction:"], ["Let's think step by step", "Response:", "Instruction:"]]
        thoughts = self.io.generate(prompt, self.num_generate_sample, stop = stop_tokens[step])

        if step == 1:
            return [y.strip() + "\n\nThe structured task plan is: " + thought.split("The structured task plan is: ")[-1].strip() for thought in thoughts if thoughts is not None]
        return [y + _ for _ in thoughts if thoughts is not None]
        
    def evaluate_thoughts(self, x, ys, step):
        """Evaluate the promise of a thought."""
        prompt = self.prompt["vote_prompt"]
        if step == 1:
            prompt += " You should only choose an option with a valid structured plan. Make sure that the structured plan of the best choice complies with the provided structure."
        prompt += f"\nAgents Description: {self.agents_desc}"
        prompt += f"\nUser Instruction: {x}"
        prompt += f"\nPlans:"
        for i, y in enumerate(ys, 1):
            prompt += f"Choice {i}: \n{y}\n"
        
        io_outputs = self.io.generate(prompt, self.num_evaluate_sample, stop=None)

        results = [0]*len(ys)
        for output in io_outputs:
            pattern = r"The best choice is:\s*(\d+)"
            match = re.search(pattern, output, re.DOTALL)
            if match:
                vote = int(match.groups()[0].replace('**', '')) - 1
                if vote in range(len(ys)):
                    results[vote] += 1
            else:
                print(f'vote no match: {[output]}')
        return results
    
    def generate(self, user_question):
        """Main search loop using a breadth-first search with pruning."""
        x = user_question
        ys = ['']
        for step in range(self.num_steps):
            new_ys = [self.generate_thoughts(x, y, step) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            values = self.evaluate_thoughts(x, new_ys, step)
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            ys = select_new_ys
        return self.evaluator.extract_answer_from_model_completion(ys[0])
