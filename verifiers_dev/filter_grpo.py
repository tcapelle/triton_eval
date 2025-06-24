import weave
from datasets import Dataset
from collections import Counter

client = weave.init("grpo-cuda/dataset-rollout")
calls = client.get_calls(
    filter={"op_names": ["weave:///grpo-cuda/dataset-rollout/op/Evaluation.predict_and_score:nLHkt0g39v37SSRn7yuZglKjboXicyHRcwrGBixaQio"],"parent_ids": ["0197a2ed-2834-7170-ac6b-dd36676fc419"]},
    query={"$expr":{"$gt":[{"$getField":"started_at"},{"$literal":1750186824.082}]}},
    sort_by=[{"field":"started_at","direction":"desc"}],
)

rows = []
is_correct_counts = Counter()

# Debug: Let's see how many calls we have
l = list(calls)
print(f"Total number of calls: {len(l)}")

if len(l) > 0:
    print(f"\nFirst call structure:")
    print(f"Has output: {hasattr(l[0], 'output') and l[0].output is not None}")
    if hasattr(l[0], 'output') and l[0].output is not None:
        print(f"Output keys: {list(l[0].output.keys()) if isinstance(l[0].output, dict) else 'Not a dict'}")
        if isinstance(l[0].output, dict) and "scores" in l[0].output:
            print(f"Scores keys: {list(l[0].output['scores'].keys()) if isinstance(l[0].output['scores'], dict) else 'Scores not a dict'}")
            print(f"Scores content: {l[0].output['scores']}")

# Now count with correct path: scores["run_scorer"]["is_correct"]
for i, call in enumerate(l):
    if call.output and "scores" in call.output:
        scores = call.output["scores"]
        if "run_scorer" in scores and isinstance(scores["run_scorer"], dict) and "is_correct" in scores["run_scorer"]:
            is_correct_value = scores["run_scorer"]["is_correct"]
            is_correct_counts[is_correct_value] += 1
            if i < 3:  # Debug first few items
                print(f"Call {i}: is_correct = {is_correct_value}")
        else:
            if i < 3:
                print(f"Call {i}: No 'run_scorer' -> 'is_correct' path found")
    else:
        if i < 3:
            print(f"Call {i}: No output or no scores in output")

print("\n========================================================")
print("Distribution of is_correct values:")
print("========================================================")
for i in range(9):  # 0 to 8
    count = is_correct_counts.get(i, 0)
    print(f"is_correct = {i}: {count} rows")

print(f"\nTotal rows processed: {sum(is_correct_counts.values())}")
print(f"Unique is_correct values found: {sorted(is_correct_counts.keys())}")

# Terminal bar chart visualization
print("\n========================================================")
print("Terminal Plot - is_correct Distribution:")
print("========================================================")

if sum(is_correct_counts.values()) > 0:
    max_count = max(is_correct_counts.values()) if is_correct_counts else 0
    scale_factor = 50 / max_count if max_count > 0 else 1  # Scale to max 50 chars width
    
    for i in range(9):  # 0 to 8
        count = is_correct_counts.get(i, 0)
        bar_length = int(count * scale_factor)
        bar = "█" * bar_length
        percentage = (count / sum(is_correct_counts.values())) * 100 if sum(is_correct_counts.values()) > 0 else 0
        print(f"is_correct={i} │{bar:<50} {count:4d} ({percentage:5.1f}%)")
    
    print(f"{'':>12}└{'─' * 50}")
    print(f"{'':>13}0{'':<48}{max_count}")
else:
    print("No data to plot")


###############

new_ds = []
for i, call in enumerate(calls):
    print(f"processing {i} of {len(calls)}")
    row = call.inputs["example"].copy()
    row["is_correct"] = call.output["scores"]["run_scorer"]["is_correct"]
    row["triton_runs"] = call.output["scores"]["run_scorer"]["triton_runs"]
    new_ds.append(row)

ds = Dataset.from_list(new_ds)
print(ds[0])

ds.filter(lambda x: x["is_correct"] < 8)
ds.save_to_disk("calls_with_is_correct.json")


# ds = Dataset.from_list(calls)
# ds.save_to_disk("calls.json")
