import weave
from weave.flow.util import async_foreach
from weave.trace.op_caller import async_call

@weave.op
async def map(ds, func, num_proc=10):
    "Apply a function asynchronously to a dataset"
    async def apply_func(row):
        "Wrapper to make the function async"
        return await async_call(func, row)
    
    results = []
    n_complete = 0
    async for _, out_row in async_foreach(ds, apply_func, max_concurrent_tasks=num_proc):
        results.append(out_row)
        n_complete += 1
        print(f"Completed {n_complete} / {len(ds)}")
    return results