import time
from pydantic import BaseModel, Field
import tiktoken
from typing import Any, Type
import weave

from agents.console import Console
from agents.tools import DEFAULT_TOOLS
from agents.tool_calling import chat_call_tool_params, perform_tool_calls


from openai import OpenAI

client = OpenAI()

AGENT_PROMPT = """
You have access to the following tools:
{tools}

You have a budget of {max_steps} steps and your model context is {max_ctx} tokens. Be careful not to exceed your budget.
"""

class AgentState(BaseModel):
    # The chat message history.
    messages: list[Any] = Field(default_factory=list)
    step_num: int = Field(default=0)

    @property
    def ctx_tokens(self):
        encoding = tiktoken.encoding_for_model("gpt-4o")
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in self.messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                else:
                    num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    def get_usage(self):
        return (f"Status: Current step: {self.step_num}, Current tokens in context: {self.ctx_tokens}\n"
                f"-------------------------------------------------------------------------------------\n\n")
    
    def insert_usage(self):
        "Append usage to last message"
        usage = self.get_usage()
        self.messages[-1]["content"] = usage + self.messages[-1]["content"]

    def get_messages_with_usage(self):
        self.insert_usage()
        return self.messages

@weave.op
def call_model(messages, model_name, temperature, tools, timeout=60, response_format=None):
    if model_name.startswith("o"):
        # o3, o4 don't support temperature
        temperature = 1.
    if response_format is None:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            timeout=timeout,
        )
    else:
        # This is the last step of the agent, so we need to parse the response, not tools needed.
        response = client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
        )
    return response

class AgentResponse(BaseModel):
    final_response: Any = Field(description="The final response from the agent, either string or validated model.")
    stop_reason: str = Field(description="The reason the agent stopped.")

class Agent(BaseModel):
    model_name: str = "gpt-4.1"
    temperature: float = 0.0
    system_message: str = "You are a helpful assistant that can help with code."
    tools: list[Any] = Field(default=DEFAULT_TOOLS)
    silent: bool = False
    response_format: Type[BaseModel] | None = Field(default=None)

    @weave.op
    def step(self, state: AgentState, response_format: BaseModel | None = None) -> AgentState:
        if not self.silent:
            Console.step_start("agent", "green")

        messages = state.get_messages_with_usage()

        if self.tools:
            tools = chat_call_tool_params(self.tools)
        else:
            tools = None

        if not self.silent:
            Console.chat_response_start()
        
        response = call_model(messages, self.model_name, self.temperature, tools, 60, response_format)

        response_message = response.choices[0].message

        if response_message.content and not self.silent:
            Console.chat_response(response_message.content)

        new_messages = []
        new_messages.append(response_message.model_dump(exclude_none=True))
        if response_message.tool_calls:
            new_messages.extend(
                perform_tool_calls(self.tools, response_message.tool_calls, self.silent)
            )

        new_history = state.messages + new_messages

        return AgentState(messages=new_history, step_num=state.step_num + 1)

    @weave.op
    def run(self, user_prompt: str, max_runtime_seconds: int = -1, max_steps: int = -1, max_ctx: int = 128_000):
        if not self.silent:
            Console.welcome(f"Using model: {self.model_name}\nTools: {self.tools}")
        agent_prompt = AGENT_PROMPT.format(tools=self.tools, max_steps=max_steps, max_ctx=max_ctx)
        messages = [{"role": "system", "content": self.system_message + agent_prompt}]
        messages.append({"role": "user", "content": user_prompt})
        state = AgentState(messages=messages, num_steps=0) 

        # Print initial user prompt if available
        if state.messages and state.messages[0]["role"] == "user":
            if not self.silent:
                Console.user_prompt(state.messages[0]["content"])
        
        start_time = time.time()
        
        while True:
            if max_steps > 0 and state.step_num >= max_steps:
                print(f"Max steps reached: {max_steps}")
                return AgentResponse(final_response=state.messages[-1]["content"], stop_reason="max_steps_reached")
            last_message = state.messages[-1]
            if last_message["role"] == "assistant" and "tool_calls" not in last_message:
                if self.response_format is not None:
                    state = self.step(state, response_format=self.response_format)
                    formatted_model = self.response_format.model_validate_json(state.messages[-1]["content"])
                    return AgentResponse(final_response=formatted_model, stop_reason="done")
                return AgentResponse(final_response=last_message["content"], stop_reason="done")
            state = self.step(state)
            if (
                max_runtime_seconds > 0
                and time.time() - start_time > max_runtime_seconds
            ):
                return AgentResponse(final_response=last_message["content"], stop_reason="time_limit_exceeded")