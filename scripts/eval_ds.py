import time
import re
import httpx
import weave
import random
import openai
from rich.console import Console
from dataclasses import dataclass
from datasets import load_dataset
import simple_parsing as sp

from my_smol_agent.tools import run_python_code

console = Console()
# server needs to be running
SERVER_URL = "http://127.0.0.1:9347"
RUN_CODE_ENDPOINT = f"{SERVER_URL}/run_code"

client = httpx.Client()
openai_client = openai.OpenAI()


@dataclass
class Args:
    dataset_name: str = "tcapelle/train_ds_triton_v2f2"
    output_dataset: str = None
    code_column: str = "format_pt_code"
    from_hub: bool = True
    debug: bool = False
    push: bool = False
    fix_imports: bool = False

args = sp.parse(Args)

if args.debug:
    weave.init('grpo-cuda/train-ds-eval')


if args.from_hub:
    console.rule(f"[bold blue]Loading dataset: {args.dataset_name}[/bold blue]")
    ds = load_dataset(args.dataset_name, split="train")
    if args.debug:
        ds = ds.select(range(10))
else:
    console.rule(f"[bold blue]Loading dataset: {args.dataset_name}[/bold blue]")
    ds = list(weave.ref(args.dataset_name).get().rows)

def send_run_request(client: httpx.Client, code: str, tests: str):
    """Sends a request to the /run_code endpoint."""
    payload = {
        "code": code,
        "tests": tests
    }
    try:
        # Use console.print for richer output
        # console.print(f"Sending request for code snippet (first 50 chars): {code[:50]}...") # Make logging less verbose
        start_time = time.monotonic()
        response = client.post(RUN_CODE_ENDPOINT, json=payload, timeout=180.0) # Adjusted timeout if needed
        end_time = time.monotonic()
        duration = end_time - start_time
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # console.print(f"Request successful (Status: [green]{response.status_code}[/green], Duration: {duration:.2f}s)") # Make logging less verbose
        return response.json(), duration # Return duration along with result
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def run_code(row):
    code_to_run, tests = split_at_tests(row[args.code_column])
    output, runtime = send_run_request(client, code_to_run, tests)
    row["pt_code_without_tests"] = code_to_run
    row["tests"] = tests
    row["stdout"] = output["stdout"]
    row["stderr"] = output["stderr"]
    row["pt_code_runs"] = output["status_code"] == 0
    row["runtime"] = runtime
    return row

def split_at_tests(code):
    """Split at ### line, it could have any number of #, at least 4 of them
    code: The code separated by ### and tests
    output: Tuple of (code, tests)
    """
    import re
    parts = re.split(r"^#{4,}", code, 1, flags=re.MULTILINE)
    if len(parts) == 1:
        return parts[0], ""
    code, tests = parts[0], parts[1]
    # remove the ```python at the beginning and end of the code
    header="import torch\ntorch.manual_seed(42)\n\nDEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    code = header + "\n\n" + code
    return code, tests


def fix_test_imports(row):
    tests = row["tests"]
    system_prompt = """You are an expert Python programmer, your task is removing duplicate lines from the code.
    Repond with the cleaned code only, nothing else. Don't put any other text or comments. Don't include ```python at the beginning or end. Code only.
    """

    user_prompt = f"Please format the following code to match the format above:\n```python\n{tests}\n```"
    response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    row["tests"] = response.choices[0].message.content
    return row

if args.debug:
    # decorate functions:
    send_run_request = weave.op(send_run_request)
    run_code = weave.op(run_code)
    split_at_tests = weave.op(split_at_tests)
    console.rule("[bold blue]Debugging[/bold blue]")
    # try split at ### line, it could have any number of #

    code_with_test = """
import torch
def add(a, b):
    return a + b
######

def test_add():
    assert add(1, 2) == 3
    return "test_add passed"

print(test_add())
"""
    code, tests = split_at_tests(code_with_test)
    console.print(f"[bold blue]code:\n{code}[/bold blue]")
    console.print(f"[bold blue]tests:\n{tests}[/bold blue]")

    # run code
    output = send_run_request(client, code, tests)
    console.print(f"[bold blue]output: {output}[/bold blue]")


    console.rule("Running on small dataset")
    rows_mapped = []
    for row in ds:
        _ = run_code(row)
        rows_mapped.append(row)

    wds = weave.Dataset(name=f"{args.dataset_name}_debug", rows=rows_mapped)
    weave.publish(wds)

if not args.debug:
    ds = ds.map(run_code, num_proc=8)


    # print some stats:
    console.print(f"[bold blue]Stats:[/bold blue]")
    console.print(f"[bold blue]Number of rows:[/bold blue] {len(ds)}")
    console.print(f"[bold blue]Number of rows that ran:[/bold blue] {sum(row['pt_code_runs'] for row in ds)}")

    if args.fix_imports:
        ds = ds.map(fix_test_imports, num_proc=20)

    if args.push:
        # push to hub
        ds.push_to_hub(args.dataset_name, commit_message="re-run-with-tests")

        # publish to weave
        weave.init('grpo-cuda/train-ds-eval')
        output_dataset = f"{args.dataset_name}" if args.output_dataset is None else args.output_dataset
        wds = weave.Dataset(name=output_dataset, rows=ds.to_list())

        weave.publish(wds)
