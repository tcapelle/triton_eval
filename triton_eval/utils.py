from weave.flow.util import async_foreach
from weave.trace.op_caller import async_call

async def map(ds, func, num_proc=10):
    "Apply a function asynchronously to a dataset"
    async def apply_func(row: dict) -> dict:
        "Wrapper to make the function async"
        return await async_call(func, row)
    
    results = []
    n_complete = 0
    async for input_row, out_row in async_foreach(ds, apply_func, max_concurrent_tasks=num_proc):
        input_row.update(out_row)
        results.append(input_row)
        n_complete += 1
        print(f"Completed {n_complete} / {len(ds)}")
    return results