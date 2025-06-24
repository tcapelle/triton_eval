import os
import httpx
import re
import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset

import weave
import openai
import simple_parsing as sp
from rich.console import Console

from triton_eval.agents.tools import remove_tests, extract_code, run_python_code
from triton_eval.kernel_checks import is_valid_kernel


from prompts import eval_system_prompt, eval_user_prompt


from prompts import eval_system_prompt, eval_user_prompt

script_dir = os.path.dirname(os.path.abspath(__file__))

console = Console()

MODEL_NAME = "qwen3-32b-grpo"




TEMPERATURE = 0.6
TIMEOUT = 60

@dataclass
class ScriptArgs:
    model_name: str = MODEL_NAME
    custom_base_url: str = "http://cw-verifiers-vllm-service:8000/v1"
    triton_server_url: str = "http://cw-verifiers-rewards-service-grpo:9347"
    num_generations: int = 1
    temperature: float = TEMPERATURE
    max_tokens: int = 12_000
    weave_project: str = "grpo-cuda/dataset-rollout"
    dataset_name: str = "tcapelle/boostrap_oai_pt_think"
    debug: bool = False
    code_token: str = sp.field(default="triton", alias="-ct")
    reasoning_token: str = sp.field(default="think", alias="-rt")

console.rule("[bold green]Running Weave Eval[/bold green]")


args = sp.parse(ScriptArgs)
print(args)

client = openai.OpenAI(
    base_url=args.custom_base_url,
)
weave.init(args.weave_project)

ds = load_dataset(args.dataset_name, split="train").to_list()
if args.debug:
    ds = ds[:3]


@weave.op
def call_triton_server(code, tests, url=args.triton_server_url):
            # Execute code on server synchronously
    with httpx.Client() as client:
        triton_endpoint = f"{url}/run_triton"
        resp = client.post(triton_endpoint, 
        json={
                    "code": code, 
                    "tests": tests,
                    "benchmark": False,
                    "benchmark_runs": 10
                },
                timeout=30.0)
        resp.raise_for_status()
        data = resp.json()


        return data

@weave.op
def call_model(
    system_prompt: str, 
    user_prompt: str, 
    model_name: str, 
    num_generations: int = 1, 
    temperature: float = 0.6, 
    max_tokens: int = 12_000,
    **model_kwargs):
    "Use reponse API for o3/o4 models, otherwise use chat completion"
    choices = []
    for i in range(num_generations):
        out = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_prompt},
                    ],
            temperature=temperature,
            max_tokens=max_tokens,
            **model_kwargs
        )
        choices.append(out.choices[0].message.content.strip())
    return choices


class OpenAICompatibleModel(weave.Model):
    "this is just a pydantic BaseModel subclass"
    model_name: str
    temperature: float
    max_tokens: int
    system_prompt: str
    user_prompt: str
    num_generations: int = 1


    @weave.op
    def predict(self, pt_code: str, entrypoint: str):
        code = remove_tests(pt_code)
        "Takes a code string and returns a response from the model"
        out = call_model(
            self.system_prompt.format(code_token=args.code_token, reasoning_token=args.reasoning_token), 
            self.user_prompt.format(pt_code=code, entrypoint=entrypoint, code_token=args.code_token, reasoning_token=args.reasoning_token), 
            self.model_name, 
            num_generations=self.num_generations,
            temperature=self.temperature, 
            max_tokens=self.max_tokens)
        return out

@weave.op
def score_one(output, tests, pt_stdout, pt_runs, pt_entrypoint):
    # get triton from model output
    triton_code = extract_code(output)

    # check valid kernel
    analysis = is_valid_kernel(triton_code, pt_entrypoint)


    # Run the triton code
    if analysis["is_valid"]:
        triton_output = call_triton_server(triton_code, tests)
        triton_runs = triton_output["status_code"] == 0
        triton_stdout = triton_output["stdout"]
        triton_stderr = triton_output["stderr"]
        is_correct = (pt_stdout == triton_stdout and pt_runs and triton_runs)
    else:
        is_correct = False
        triton_runs = False
        triton_stdout = ""
        triton_stderr = ""
    result = {
        "is_valid": analysis["is_valid"],
        "triton_runs": triton_runs,
        "triton_stdout": triton_stdout,
        "triton_stderr": triton_stderr,
        "is_correct": is_correct,
        "validity": analysis["reason"]
    }

    return result

def aggregate_results(results):
    return {
        "is_valid": sum([1 for result in results if result["is_valid"]]),
        "triton_runs": sum([1 for result in results if result["triton_runs"]]),
        "triton_stdout": sum([1 for result in results if result["triton_stdout"]]),
        "triton_stderr": sum([1 for result in results if result["triton_stderr"]]),
        "is_correct": sum([1 for result in results if result["is_correct"]]),
        "validity": sum([1 for result in results if result["validity"]]),
    }

@weave.op
def run_scorer(output, tests, pt_stdout, pt_runs, pt_entrypoint):
    "Runs the code and returns the output"
    results = []
    for choice in output:
        result = score_one(choice, tests, pt_stdout, pt_runs, pt_entrypoint)
        results.append(result)
    return aggregate_results(results)

scorers = [run_scorer]

weave_model = OpenAICompatibleModel(
    model_name=args.model_name,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    system_prompt=eval_system_prompt,
    user_prompt=eval_user_prompt,
    num_generations=args.num_generations,
)

evaluation = weave.Evaluation(dataset=ds, scorers=scorers, evaluation_name=args.model_name)

asyncio.run(evaluation.evaluate(model=weave_model))