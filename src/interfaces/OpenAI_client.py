import os
import time
from openai import OpenAI

class OpenAIClient:
    """
    A client wrapper for interacting with OpenAI chat models (e.g., GPT-4o).

    This class handles API initialization, prompt generation, retry logic, and 
    supports both single and multiple completions.

    Attributes:
        api_key (str): OpenAI API key.
        model_ckpt (str): The model checkpoint to use (e.g., 'gpt-4o-mini').
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Not currently supported in OpenAI API but included for compatibility.
        top_p (float): Nucleus sampling probability.
        stop (List[str]): Stop sequences for generation.
    """

    def __init__(
        self,
        model_name="gpt-4o-mini",
        tokenizer_name = None
    ):
        """
        Initializes the LLMClient with specified generation parameters.

        Args:
            api_key (str, optional): OpenAI API key. Defaults to environment variable OPENAI_API_KEY.
            model_ckpt (str): Model checkpoint to use.
            max_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): (Reserved for compatibility) Top-k sampling parameter.
            top_p (float): Nucleus sampling parameter.
            stop (List[str], optional): Stop sequences.
        """

        self.api_key = None or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise Exception("OpenAI api key not set. Set API key using export OPENAI_API_KEY=<api key value>")

        self.model_name = model_name

    def generate(self, prompt, **generation_kwargs):
        """
        Generate a single response from the LLM.

        Args:
            prompt (str): User prompt to send to the model.

        Returns:
            str: Model-generated text response.
        """

        messages = [{"role": "user", "content": prompt}]

        ans, timeout = "", 5
        while not ans:
            try:
                time.sleep(timeout)
                completion = self.client.chat.completions.create(model=self.model_name, messages=messages, **generation_kwargs)
                ans = completion.choices[0].message.content
            except Exception as e:
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
        return ans

    def generate_n(self, prompt, n=1, **generation_kwargs):
        """
        Generate multiple responses from the LLM.

        Args:
            prompt (str): User prompt to send to the model.
            n (int): Number of completions to generate.

        Returns:
            List[str]: List of generated text responses.
        """
        
        messages = [{"role": "user", "content": prompt}]

        ans, timeout = [], 5
        while not ans:
            try:
                time.sleep(timeout)
                completion = self.client.chat.completions.create(model=self.model_name, messages=messages, n=n, **generation_kwargs)
                ans = [choice.message.content for choice in completion.choices]
            except Exception as e:
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
        return ans

    def close(self):
        pass