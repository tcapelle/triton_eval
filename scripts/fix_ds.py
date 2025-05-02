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
    revision: str = "cd4246a288306175a31d49d1f7f1d8bde47b87a4"
    output_dataset: str = None
    code_column: str = "format_pt_code"
    from_hub: bool = True
    debug: bool = False
    push: bool = False
    fix_imports: bool = False
    fix_entrypoint: bool = False
    split_at_tests: bool = False
    run_code: bool = False

args = sp.parse(Args)

if args.debug:
    weave.init('grpo-cuda/train-ds-eval')


if args.from_hub:
    console.rule(f"[bold blue]Loading dataset: {args.dataset_name}[/bold blue]")
    ds = load_dataset(args.dataset_name, split="train", revision=args.revision)
    if args.debug:
        ds = ds.select(range(5))
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
    code_to_run = row["pt_code_without_tests"]
    tests = row["tests"]
    output, runtime = send_run_request(client, code_to_run, tests)
    row["stdout"] = output["stdout"]
    row["stderr"] = output["stderr"]
    row["pt_code_runs"] = output["status_code"] == 0
    row["runtime"] = runtime
    return row

def split_at_tests(row):
    """Split at ### line, it could have any number of #, at least 4 of them
    code: The code separated by ### and tests
    output: Tuple of (code, tests)
    """

    code = row[args.code_column]
    parts = re.split(r"^#{4,}", code, 1, flags=re.MULTILINE)
    if len(parts) == 1:
        return parts[0], ""
    code, tests = parts[0], parts[1]
    # remove the ```python at the beginning and end of the code
    header_imports = ["import torch\n", "torch.manual_seed(42)\n", "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"]
    header = ""
    for import_line in header_imports:
        if import_line in tests:
            continue
        else:
            header += import_line
    tests = header + "\n" + tests
    row["tests"] = tests
    row["pt_code_without_tests"] = code
    return row


def fix_entrypoint(row):
    """We want to replace the entrypoint name by removing any mention of torch or pt"""
    entrypoint = row["entrypoint"]

    # we need a list that is ordered by substrings, eg. search pytorch instead of torch
    replacements = [
        ("_pytorch", "_func"),
        ("pytorch_", "func_"),
        ("_torch", "_func"),
        ("torch_", "func_"),
        ("_pt", "_func"),
        ("pt_", "func_")
    ]
    
    new_entrypoint = entrypoint
    for old, new in replacements:
        new_entrypoint = new_entrypoint.replace(old, new)
    
    # Only replace "torch" or "pt" as whole words, not as part of other words
    # This prevents replacing "torch" in contexts like "import torch" or "torch.sum"
    if re.search(r'\btorch\b', new_entrypoint):
        new_entrypoint = re.sub(r'\btorch\b', 'func', new_entrypoint)
    if re.search(r'\bpt\b', new_entrypoint):
        new_entrypoint = re.sub(r'\bpt\b', 'func', new_entrypoint)
    
    # Update the entrypoint field in the row
    row["entrypoint"] = new_entrypoint
    
    # we also need to update the code and tests, but be careful not to replace torch in function calls
    # like torch.sum or imports like from torch.nn import ...
    code = row[args.code_column]
    
    # Use word boundaries to only replace the exact function name
    pattern = r'\b' + re.escape(entrypoint) + r'\b'
    # Also create a pattern for the test function name
    test_pattern = r'\btest_' + re.escape(entrypoint) + r'\b'
    new_test_name = 'test_' + new_entrypoint

    # Perform both replacements on the code string
    code_fixed_entrypoint = re.sub(pattern, new_entrypoint, code)
    code_fixed_tests = re.sub(test_pattern, new_test_name, code_fixed_entrypoint)

    row[args.code_column] = code_fixed_tests
    console.print(f"[bold blue]Fixed entrypoint: {entrypoint} -> {new_entrypoint}[/bold blue]")
    return row



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


# filters to check dataset correctness
def _get_test_signature(test_code):
    # use a re to find the test_* items
    return re.findall(r"test_[^ ]+", test_code)

def check_tests_signature(row):
    "We should have a test that match the entrypoint -> test_<entrypoint>"
    tests = row["tests"]
    if tests == "":
        console.print("[bold red]Error: tests is empty[/bold red]")
        return False
    entrypoint = row["entrypoint"]
    if f"test_{entrypoint}" not in tests:
        actual_tests = _get_test_signature(tests)
        console.print(f"[bold red]Error: test_{entrypoint} not in tests[/bold red]")
        console.print(f"[bold red]Actual tests: {actual_tests}[/bold red]")
        return False
    return True

def check_entrypoint_name(row):
    "We should not have pt or torch in the entrypoint name"
    entrypoint = row["entrypoint"]
    # Use regex with word boundaries to check for standalone 'pt' or 'torch'
    if re.search(r'\bpt\b', entrypoint) or re.search(r'\btorch\b', entrypoint):
        console.print(f"[bold red]Error: entrypoint {entrypoint} contains standalone pt or torch[/bold red]")
        return False
    return True


if args.debug:
    # decorate functions:
    send_run_request = weave.op(send_run_request)
    run_code = weave.op(run_code)
    split_at_tests = weave.op(split_at_tests)
    fix_test_imports = weave.op(fix_test_imports)
    fix_entrypoint = weave.op(fix_entrypoint)

    console.rule("[bold blue]Debugging[/bold blue]")

    # Print initial entrypoints for debug
    console.rule("[bold blue]Initial Entrypoints (Debug)[/bold blue]")
    for i, row_data in enumerate(ds):
        # Check if ds is a datasets.Dataset or a list
        if hasattr(ds, 'features'): # Likely a datasets.Dataset
            entrypoint = row_data.get('entrypoint', 'N/A')
        else: # Likely a list of dicts
            entrypoint = row_data.get('entrypoint', 'N/A')
        console.print(f"Row {i} Initial Entrypoint: {entrypoint}")

    console.rule("Running on small dataset")
    rows_mapped = ds.to_list()

    if args.fix_entrypoint:
        console.rule("Fixing entrypoint")
        temp_rows = []
        for row in rows_mapped:
            temp_rows.append(fix_entrypoint(row))
        rows_mapped = temp_rows

    if args.split_at_tests:
        console.rule("Splitting at tests")
        temp_rows = []
        for row in rows_mapped:
            temp_rows.append(split_at_tests(row))
        rows_mapped = temp_rows


    if args.run_code:
        console.rule("Running code")
        temp_rows = []
        for row in rows_mapped:
            temp_rows.append(run_code(row))
        rows_mapped = temp_rows

    if args.fix_imports:
        console.rule("Fixing imports")
        temp_rows = []
        for row in rows_mapped:
            temp_rows.append(fix_test_imports(row))
        rows_mapped = temp_rows


    wds = weave.Dataset(name=f"{args.dataset_name}_debug", rows=rows_mapped)
    weave.publish(wds)

    # Add checks similar to the non-debug path
    console.rule("[bold blue]Running Checks on Debug Dataset[/bold blue]")
    checks_pass_debug = True
    failed_signature_check = [row for row in rows_mapped if not check_tests_signature(row)]
    console.print(f"Number of rows that failed tests signature check (debug): {len(failed_signature_check)}")
    if len(failed_signature_check) > 0:
        checks_pass_debug = False

    failed_entrypoint_check = [row for row in rows_mapped if not check_entrypoint_name(row)]
    console.print(f"Number of rows that failed entrypoint name check (debug): {len(failed_entrypoint_check)}")
    if len(failed_entrypoint_check) > 0:
        checks_pass_debug = False

    if checks_pass_debug:
        console.print("[green]All debug checks passed![/green]")
    else:
        console.print("[red]Some debug checks failed.[/red]")


if not args.debug:


    if args.fix_entrypoint:
        ds = ds.map(fix_entrypoint, num_proc=8)

    if args.split_at_tests:
        ds = ds.map(split_at_tests, num_proc=8)

    if args.run_code:
        ds = ds.map(run_code, num_proc=8)


    # print some stats:
    console.print(f"[bold blue]Stats:[/bold blue]")
    console.print(f"[bold blue]Number of rows:[/bold blue] {len(ds)}")
    console.print(f"[bold blue]Number of rows that ran:[/bold blue] {sum(row['pt_code_runs'] for row in ds)}")

    if args.fix_imports:
        ds = ds.map(fix_test_imports, num_proc=20)

    checks_pass = True
    fds = ds.filter(lambda x: check_tests_signature(x))
    diff = len(ds) - len(fds)
    console.print(f"Number of rows that failed tests signature check: {diff}")
    if diff > 0:
        checks_pass = False

    ffds = fds.filter(lambda x: check_entrypoint_name(x))
    diff = len(ffds) - len(fds)
    console.print(f"Number of rows that failed entrypoint name check: {diff}")
    if diff > 0:
        checks_pass = False

    ds = ffds
    
    if args.push:
        # push to hub
        ds.push_to_hub(args.dataset_name, commit_message="fixed")

        # publish to weave
        weave.init('grpo-cuda/train-ds-eval')
        output_dataset = f"{args.dataset_name}" if args.output_dataset is None else args.output_dataset
        wds = weave.Dataset(name=output_dataset, rows=ds.to_list())

        weave.publish(wds)
