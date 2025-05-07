from dataclasses import dataclass

import openai
import simple_parsing as sp

@dataclass
class ScriptArgs:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    ip: str = "127.0.0.1"
    port: int = 8000

args = sp.parse(ScriptArgs)

openai_client = openai.OpenAI(
    base_url=f"http://{args.ip}:{args.port}/v1",
    api_key="NoKey",
)

def make_call(prompt: str):
    response = openai_client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

print(f"Making call to: {args.model_name} at {args.ip}:{args.port}")
out = make_call("Hello, world!")
print(f"output: {out}")

