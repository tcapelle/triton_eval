# axolotl_dev/test_server.py
from datasets import load_dataset
import httpx
import asyncio
import os
import time # Import time
import statistics # Import statistics
from rich.console import Console # Import rich console
from rich.rule import Rule       # Import rich rule
import sys # Import sys for exiting
import traceback # Import traceback for better error printing
import simple_parsing as sp # Import simple_parsing
from dataclasses import dataclass
import multiprocessing # Ensure multiprocessing is imported


# Initialize rich console
console = Console()

# Server configuration
SERVER_URL = os.environ.get("TRITON_SERVER_URL", "http://127.0.0.1:9347")
RUN_CODE_ENDPOINT = f"{SERVER_URL}/run_code"
START_WORKERS_ENDPOINT = f"{SERVER_URL}/start_workers" # New endpoint URL
STOP_WORKERS_ENDPOINT = f"{SERVER_URL}/stop_workers"   # New endpoint URL

# CODE_COLUMN_NAME = "triton_code"
CODE_COLUMN_NAME = "pt_code_without_tests"

# Dataset configuration
ds_name = "tcapelle/train_ds_triton"
NUM_TEST_EXAMPLES = 100 # Reduced for quicker testing of start/stop

# --- Argument Definition using Dataclass ---
@dataclass
class TestArgs:
    """Command-line arguments for the test script."""
    start_index: int = 0 # The starting index (0-based) of the dataset examples to test.
    num_examples: int = NUM_TEST_EXAMPLES # The number of examples to test starting from start-index

# -----------------------------------------

async def send_run_request(client: httpx.AsyncClient, code: str, tests: str):
    # ... (send_run_request function remains the same) ...
    """Sends a request to the /run_code endpoint."""
    payload = {
        "code": code,
        "tests": tests
    }
    try:
        # Use console.print for richer output
        # console.print(f"Sending request for code snippet (first 50 chars): {code[:50]}...") # Make logging less verbose
        start_time = time.monotonic()
        response = await client.post(RUN_CODE_ENDPOINT, json=payload, timeout=180.0) # Adjusted timeout if needed
        end_time = time.monotonic()
        duration = end_time - start_time
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # console.print(f"Request successful (Status: [green]{response.status_code}[/green], Duration: {duration:.2f}s)") # Make logging less verbose
        return response.json(), duration # Return duration along with result
    except httpx.TimeoutException as e:
        # --- Catch client-side timeouts specifically ---
        console.print(f"[bold red]Client Timeout Error:[/bold red] Request timed out after configured duration ({client.timeout.read:.1f}s). ({type(e).__name__})")
        return None, None
    except httpx.ConnectError as e:
        # --- Improved Error Reporting ---
        console.print(f"[bold red]Connect Error:[/bold red] Connection refused. Is the server running at {SERVER_URL}? ({type(e).__name__})")
        return None, None # Return None for duration on failure
    except httpx.HTTPStatusError as e:
         # Specifically handle 503 (workers not running or queue full) and 504 (timeout)
        if e.response.status_code == 503:
             console.print(f"[bold yellow]HTTP Error 503:[/bold yellow] Service Unavailable - {e.response.text} ({type(e).__name__})")
        elif e.response.status_code == 504:
             console.print(f"[bold red]HTTP Error 504:[/bold red] Gateway Timeout - {e.response.text} ({type(e).__name__})")
        else:
             console.print(f"[bold red]HTTP Error:[/bold red] {e.response.status_code} - {e.response.text} ({type(e).__name__})")
        return None, None # Return None for duration on failure
    except httpx.RequestError as e:
        console.print(f"[bold red]Request Error:[/bold red] {e} ({type(e).__name__})\nRequest: {e.request}")
        return None, None # Return None for duration on failure
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during send_run_request:[/bold red] {e} ({type(e).__name__})")
        return None, None # Return None for duration on failure


async def manage_workers(client: httpx.AsyncClient, action: str):
    """Starts or stops the workers via API call."""
    endpoint = START_WORKERS_ENDPOINT if action == "start" else STOP_WORKERS_ENDPOINT
    console.print(f"Attempting to [cyan]{action}[/cyan] workers at {endpoint}...")
    try:
        response = await client.post(endpoint, timeout=30.0) # Timeout for worker management
        response.raise_for_status()
        console.print(f"[green]Successfully {action}ed workers:[/green] {response.json().get('message')}")
        return True
    except httpx.ConnectError:
        console.print(f"[bold red]Connect Error:[/bold red] Failed to connect to server at {SERVER_URL} to {action} workers.")
        return False
    except httpx.HTTPStatusError as e:
        # Handle conflicts gracefully (e.g., trying to start when already started)
        if e.response.status_code == 409:
             console.print(f"[yellow]Conflict:[/yellow] Could not {action} workers - {e.response.text}")
             # If starting failed due to conflict, assume they are running for the test to proceed
             return action == "start" # Return True if starting, False if stopping
        else:
             console.print(f"[bold red]HTTP Error:[/bold red] Failed to {action} workers ({e.response.status_code}): {e.response.text}")
        return False
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during worker management ({action}):[/bold red] {e} ({type(e).__name__})")
        return False


async def main(args: TestArgs): # Accept TestArgs instance
    """Loads data for a specific range, starts workers, sends requests, stops workers."""
    console.print(Rule("[bold blue]Starting Server Test[/bold blue]"))
    console.print(f"Targeting dataset range: index {args.start_index} to {args.start_index + args.num_examples - 1}")

    async with httpx.AsyncClient() as client:
        if not await manage_workers(client, "start"):
            console.print("[bold red]Failed to start workers. Aborting test.[/bold red]")
            return

        try:
            console.print(f"Loading dataset: [cyan]{ds_name}[/cyan]")
            try:
                ds = load_dataset(ds_name, split='train', streaming=True)
                ds_iterator = iter(ds.skip(args.start_index).take(args.num_examples))
            except Exception as e:
                console.print(f"[bold red]Error loading or processing dataset range:[/bold red] {e}")
                return

            console.print(f"Testing server at: [link={SERVER_URL}]{RUN_CODE_ENDPOINT}[/link]")
            tasks = []
            console.print(f"Preparing {args.num_examples} requests from specified range...")
            current_original_index = args.start_index
            original_indices_in_batch = [] # Store original indices corresponding to tasks

            for example in ds_iterator:
                code = example.get(CODE_COLUMN_NAME)
                tests = example.get("tests")
                if code and tests:
                    tasks.append(send_run_request(client, code, tests))
                    original_indices_in_batch.append(current_original_index)
                else:
                    console.print(f"[yellow]Warning:[/yellow] Skipping example at original index {current_original_index} due to missing '{CODE_COLUMN_NAME}' or 'tests' field.")
                current_original_index += 1

            actual_num_tasks = len(tasks)
            if not tasks:
                console.print("[bold red]No valid examples found in the specified range to send requests.[/bold red]")
                return

            console.print(Rule("[bold blue]Sending Requests[/bold blue]"))
            console.print(f"Sending [bold yellow]{actual_num_tasks}[/bold yellow] requests concurrently...")
            overall_start_time = time.monotonic()
            results_with_durations = await asyncio.gather(*tasks, return_exceptions=True)
            overall_end_time = time.monotonic()
            total_duration = overall_end_time - overall_start_time
            console.print(f"Request batch finished in {total_duration:.2f}s.")

            console.print(Rule("[bold blue]Processing Results[/bold blue]"))
            successful_executions = 0
            execution_errors = 0
            request_failures = 0
            successful_durations = []
            execution_error_original_indices = []
            request_failure_original_indices = []
            other_gather_errors = 0

            for i, res_or_exc in enumerate(results_with_durations):
                original_index = original_indices_in_batch[i] # Get original index for this task
                task_report_index = i + 1

                if isinstance(res_or_exc, Exception):
                     console.print(f"Result {task_report_index} (Original Index {original_index}): [bold red]Gather Error:[/bold red] {res_or_exc}")
                     request_failures += 1
                     request_failure_original_indices.append(original_index)
                     other_gather_errors +=1
                     continue

                result, duration = res_or_exc
                console.print(f"Result {task_report_index} (Original Index {original_index}):", style="bold")
                if result and duration is not None:
                     status_code = result.get('status_code')
                     stderr_content = str(result.get('stderr', ''))
                     if status_code == 0:
                         successful_executions += 1
                         successful_durations.append(duration)
                         status_style = "green"
                         console.print(f"  Status Code: [{status_style}]{status_code}[/{status_style}]")
                         console.print(f"  Stdout (first 100 chars): {str(result.get('stdout'))[:100]}...")
                     else:
                         execution_errors += 1
                         execution_error_original_indices.append(original_index)
                         status_style = "red"
                         console.print(f"  Status Code: [{status_style}]{status_code}[/{status_style}]")
                         console.print(f"  Stdout (first 100 chars): {str(result.get('stdout'))[:100]}...")
                         if stderr_content:
                              console.print(f"  [bold red]Stderr:[/bold red]\n{stderr_content}")
                         else:
                              console.print("  [bold red]Stderr:[/bold red] (Empty)")
                else:
                     request_failures += 1
                     request_failure_original_indices.append(original_index)
                     console.print("  [yellow]Request failed or server unavailable (reported by send_run_request).[/yellow]")

            console.print(Rule("[bold blue]Summary[/bold blue]"))
            console.print(f"Dataset Range Tested: Indices {args.start_index} to {args.start_index + actual_num_tasks - 1} (attempted {args.num_examples})")
            console.print(f"Total examples processed in batch: {actual_num_tasks}")
            console.print(f"Successful executions (status_code == 0): [green]{successful_executions}[/green]")
            console.print(f"Execution errors (status_code != 0): [yellow]{execution_errors}[/yellow]")
            console.print(f"Request failures (HTTP/network/timeout): [red]{request_failures}[/red]")
            if other_gather_errors > 0:
                 console.print(f"  (Includes {other_gather_errors} unexpected errors during request gathering)")
            console.print(f"Total time for request batch execution: {total_duration:.2f}s")

            if execution_error_original_indices:
                 console.print(f"Original dataset indices of execution errors: {sorted(execution_error_original_indices)}")
            if request_failure_original_indices:
                 console.print(f"Original dataset indices of request failures: {sorted(request_failure_original_indices)}")

            if successful_durations:
                mean_duration = statistics.mean(successful_durations)
                stdev_duration = statistics.stdev(successful_durations) if len(successful_durations) > 1 else 0.0
                console.print(f"Mean time per successful execution: {mean_duration:.3f}s")
                console.print(f"Std dev time per successful execution: {stdev_duration:.3f}s")
            elif successful_executions == 0 and actual_num_tasks > 0:
                console.print("No successful executions to calculate timing statistics.")

        finally:
            console.print(Rule("[bold blue]Stopping Workers[/bold blue]"))
            await manage_workers(client, "stop")

    console.print(Rule("[bold blue]Test Finished[/bold blue]"))

if __name__ == "__main__":
    args = sp.parse(TestArgs) # Extract the TestArgs instance

    # --- Argument Validation ---
    if args.start_index < 0:
        console.print("[bold red]Error: --start-index cannot be negative.[/bold red]")
        sys.exit(1)
    if args.num_examples <= 0:
        console.print("[bold red]Error: --num-examples must be positive.[/bold red]")
        sys.exit(1)
    # --- End Validation ---

    # Ensure simple_parsing is installed
    try:
        import simple_parsing
    except ImportError:
        console.print("[bold red]Error: simple_parsing is not installed. Please run 'pip install simple_parsing'[/bold red]")
        sys.exit(1)

    try:
        # Pass the TestArgs instance to main
        asyncio.run(main(args=args))
    except Exception as e:
        console.print(f"[bold red]Unhandled error in main test execution:[/bold red] {e}")
        traceback.print_exc()
        sys.exit(1)
