"""Batch reward tester.

Loads `tcapelle/train_ds_triton` in streaming mode, selects a range of examples,
starts/stops the Triton worker pool via HTTP, and computes `reward_code_runs`
concurrently across that batch.

Run with (examples):

    python -m axolotl_dev.test_code_reward --start-index 0 --num-examples 10
"""

import asyncio
import os
from dataclasses import dataclass

import httpx
from datasets import load_dataset
from rich import print
import simple_parsing as sp

from rewards import reward_code_runs

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    start_index: int = 0
    num_examples: int = 5

# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("TRITON_SERVER_URL", "http://127.0.0.1:9347")
START_WORKERS_ENDPOINT = f"{SERVER_URL}/start_workers"
STOP_WORKERS_ENDPOINT = f"{SERVER_URL}/stop_workers"


async def _manage_workers(action: str) -> None:
    url = START_WORKERS_ENDPOINT if action == "start" else STOP_WORKERS_ENDPOINT
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, timeout=30.0)
            if resp.status_code in (200, 409):
                print(f"[green]{action.title()} workers:[/green] {resp.json().get('message', resp.text)}")
            else:
                print(f"[red]Failed to {action} workers:[/red] {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"[red]Error trying to {action} workers:[/red] {exc}")


# ---------------------------------------------------------------------------
# Build completions and run rewards
# ---------------------------------------------------------------------------


async def main(args: Args) -> None:
    await _manage_workers("start")

    try:
        ds = load_dataset("tcapelle/train_ds_triton", split="train", streaming=True)
        ds_it = iter(ds.skip(args.start_index).take(args.num_examples))

        completions = []
        tests_list = []
        pt_outputs = []

        for ex in ds_it:
            code = ex.get("pt_code_without_tests")
            tests = ex.get("tests")
            pt_out = ex.get("pytorch_code_output", "")

            if not code or not tests:
                print("[yellow]Skipping example with missing fields.[/yellow]")
                continue

            content = f"<think>auto</think>\n```python\n{code}\n{tests}\n```"
            completions.append([{ "content": content }])
            tests_list.append(tests)
            pt_outputs.append(pt_out)

        # Compute rewards concurrently using thread pool since reward_code_runs is synchronous
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None,
                reward_code_runs,
                [comp],
                [tests_list[i]],
                [pt_outputs[i] if pt_outputs[i] else ""],
            )
            for i, comp in enumerate(completions)
        ]

        batch_rewards = await asyncio.gather(*tasks)
        print("Computed rewards for", len(batch_rewards), "samples")
        for idx, r in enumerate(batch_rewards):
            print(f"Sample {idx + args.start_index}:", r)

    finally:
        await _manage_workers("stop")


if __name__ == "__main__":
    cli_args = sp.parse(Args)
    asyncio.run(main(cli_args))