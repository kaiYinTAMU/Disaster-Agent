import re
from typing import Dict, List, Tuple

from interfaces.IO_Interface import IO_Interface
from utils.agent_utils import concat_all_parent_steps, concat_direct_answers, concat_subquestions_and_subanswers, concat_subquestions_and_subanswers_as_da, reach_terminal_subquestion
from prompts.prompt_builder import PromptBuilder
from evaluators.evaluators import Evaluator

class MCTS_Generator:
    """Generator generates children nodes"""

    def __init__(self, args, model_name, tokenizer_name) -> None:
        generation_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        if args.api == "huggingface":
            generation_kwargs["top_k"] = args.top_k
            
        self.io = IO_Interface(args.api, model_name, tokenizer_name, generation_kwargs)        
        self.evaluator = Evaluator()

        self.num_divide_and_conquer = args.num_divide_and_conquer
        self.num_direct_answer = args.num_direct_answer  
        self.num_divide_and_conquer_votes = args.num_divide_and_conquer_votes       
        self.num_chain_of_thought = args.num_chain_of_thought   
        
        self.promptbuilder = PromptBuilder(args.prompts_dir, args.tools_dir, args.use_fewshot)
        self.question_index = 3

    def generate_direct_answer(self, user_question: str, solution_trace: Dict[int, Dict[str, str]], paraphrased: bool, parent_is_subquestion: bool, parent_is_self_refine=False):
        direct_answer_list = []
        # For Self Refine
        if parent_is_self_refine:
            existing_direct_answer = solution_trace[0]['answers'][-1][1] + "\n"
            matches = re.findall(r'step \d+:', existing_direct_answer)
            next_answer_id = len(matches) + 1
        else:
            # For Divide and Conquer
            if parent_is_subquestion:
                existing_direct_answer, next_answer_id = concat_subquestions_and_subanswers_as_da(solution_trace)
            else:
                # For Root, Direct Answer, Rephrase and COT
                existing_direct_answer, next_answer_id = concat_direct_answers(solution_trace)
        
        io_input = self.promptbuilder.build_direct_answer_prompt(user_question, existing_direct_answer, next_answer_id)
        io_output_list = self.io.generate(
            model_input=io_input,
            num_return=self.num_direct_answer,
            stop = [f"step {next_answer_id + 1}:"]
        )
        
        for x in io_output_list:
            if x.strip() not in direct_answer_list and len(x.strip())!=0:
                direct_answer_list.append(x.strip())

        # TODO: new added
        if len(direct_answer_list) == 0:
            direct_answer_list = [io_output_list[0].strip()]

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  
        potential_answers_list = [None] * len(direct_answer_list)

        return direct_answer_list, potential_answers_list

    def _get_most_likely_answer(self, user_question, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1.0
        else:  
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                user_question, 
                io_output_list, 
                self.promptbuilder,
                self.io
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):  
        io_input = self.promptbuilder.build_chain_of_thought_prompt(question, paraphrased, hint)
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  
        return io_input, cleaned_io_output_list

    def generate_chain_of_thought(self, user_question: str, paraphrased: bool, hint: str):
        chain_of_thought_list, value_list = [], []

        num_return = self.num_chain_of_thought
        _, io_output_list = self._fewshot_cot_answer_question(
            question=user_question, 
            paraphrased=paraphrased, 
            num_return=num_return, 
            hint=hint
        )
               
        most_likely_answer, likelihood = self._get_most_likely_answer(user_question, io_output_list)            

        chain_of_thought_list.append(most_likely_answer)
        value_list.append(likelihood)
 
        return chain_of_thought_list, value_list

    def generate_subquestions(self, user_question, solution_trace, paraphrased):
        subquestion_list, subanswer_list, value_list = [], [], []

        # concatenates only all the subquestions and subanswers in the prompt format
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subquestions_and_subanswers(
            solution_trace, self.question_index
        )

        io_input = self.promptbuilder.build_divide_and_conquer_prompt(user_question, existing_subquestions_and_subanswers, self.question_index, next_subquestion_id)
        
        io_output_list = self.io.generate(
            io_input,
            num_return=self.num_divide_and_conquer,
            stop=[f"Answer {self.question_index}.{next_subquestion_id}",],
        )

        subquestion_list = [o.split(f"Question {self.question_index}.{next_subquestion_id}:")[-1].strip() for o in io_output_list if o is not None]

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = self.promptbuilder.build_divide_and_conquer_prompt(user_question, existing_subquestions_and_subanswers, self.question_index, next_subquestion_id, subquestion)
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.num_chain_of_thought
        else:
            num_return = self.num_divide_and_conquer_votes
        
        io_output_list = self.io.generate(
            io_input_list,
            num_return=num_return,
            stop=[f"Question {self.question_index}.{next_subquestion_id + 1}"],
        )

        # creates one group of subanswers against every subquestion candidate
        if len(io_output_list) > 1:
            cleaned_io_output_list = [
                [io_output.split(f"Answer {self.question_index}.{next_subquestion_id}:")[-1].strip() for io_output in io_output_group] for io_output_group in io_output_list
            ]

            for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
                try:  
                    most_likely_answer, likelihood = self._get_most_likely_answer(user_question, cleaned_io_output_group) # find most likely answer from every group
                except Exception as e:
                    print("Error")

                subanswer_list.append(most_likely_answer)
                value_list.append(likelihood)
        else:
            subanswer_list = [io_output_list[0].strip()]
            value_list = [1.0]

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list

    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.promptbuilder.build_rephrase_prompt(user_question)
        io_output = self.io.generate(model_input=io_input, num_return=1)[0]

        io_output = io_output.split("?")[0] + "?"
        io_output = "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  
        potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list
    
    def generate_self_refine(self, user_question, solution_trace):
        self_reflection_and_refine_list = []
        existing_steps, next_step = concat_all_parent_steps(solution_trace)
        
        io_input = self.promptbuilder.build_self_refine_prompt(user_question=user_question, existing_steps=existing_steps, type="reflect")
        
        io_output_list = self.io.generate(
            model_input=io_input, num_return=3, stop=["<REFINE>", "</REFLECT>", f"step {next_step}"] 
        )

        io_output_reflect = []
        for io_output in io_output_list:
            feedback = io_output.strip().split("<REFLECT>")[-1].strip()
            if "All the above steps are correct" in feedback:
                io_output_reflect.append(existing_steps.strip())
            else:
                io_input = self.promptbuilder.build_self_refine_prompt(user_question=user_question, existing_steps = existing_steps, type="refine", feedback=feedback)
                io_output = self.io.generate(model_input=io_input, num_return=1, stop=["</REFINE>", f"step {next_step}"])[0].strip()
                io_output_reflect.append(io_output.split("<REFINE>")[-1].strip())
        
        if io_output_reflect:
            outputs = io_output_reflect
        else:
            outputs = [existing_steps.strip()]
        
        potential_answers_list = [None] * len(outputs)
        return outputs, potential_answers_list

    def close(self):
        """
        Gracefully release GPU and model resources used by MCTS_Generator.
        """
        try:
            if hasattr(self.io, "close"):
                self.io.close()
        except Exception as e:
            print(f"[Warning] Failed to close IO_Interface: {e}")

        print("[MCTS_Generator] Cleanup completed successfully.")
