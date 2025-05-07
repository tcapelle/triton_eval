import openai
from prompts import formatting_prompt, PytorchCodeWithTestCases
from datasets import load_from_disk

ds = load_from_disk("train_ds_triton_v2")

client = openai.OpenAI()

user_prompt = "Here is the pytorch code to format: {code}"

def format_pt_code(row):
    try:
        code = row["pytorch_code_with_tests"]
        response = client.beta.chat.completions.parse(
            model="o4-mini",
            messages=[{
                "role": "user", "content": formatting_prompt + "\n\n" + user_prompt.format(code=code)
            }],
            response_format=PytorchCodeWithTestCases,
        )
        return {"format_pt_code": response.choices[0].message.parsed.pytorch_code_with_test_cases, 
                "entrypoint": response.choices[0].message.parsed.entrypoint}
    except Exception as e:
        print(f"Error formatting code: {e}")
        return {"format_pt_code": None, "entrypoint": None}

# filter out non rows

def empty_code(row):
    pt_code = row["pytorch_code_with_tests"] 
    if pt_code is None:
        return False
    pt_code = pt_code.strip()
    if pt_code == "":
        return False
    return True

fds = ds.filter(empty_code)
print(f"Removed {len(ds) - len(fds)} rows")


formatted_ds = fds.map(format_pt_code, num_proc=20)


def print_side_by_side(row):
    print(f"--------- {row['entrypoint']} ---------\n-----------------------------------")
    print(row["pytorch_code_with_tests"])
    print("-"*100)
    print(row["format_pt_code"])

print_side_by_side(formatted_ds[1])

formatted_ds.save_to_disk("train_ds_triton_v2f")