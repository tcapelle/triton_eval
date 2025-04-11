import time
from pydantic import BaseModel, Field
from typing import Any, Type
import weave

from my_smol_agent.console import Console
from my_smol_agent.tools import DEFAULT_TOOLS
from my_smol_agent.tool_calling import chat_call_tool_params, perform_tool_calls
import litellm

class AgentState(BaseModel):
    # The chat message history.
    messages: list[Any] = Field(default_factory=list)
    num_steps: int = Field(default=0)
    max_ctx: int = Field(default=128_000)

class AgentResponse(BaseModel):
    final_response: Any = Field(description="The final response from the agent, either string or validated model.")
    stop_reason: str = Field(description="The reason the agent stopped.")

class Agent(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    system_message: str = "You are a helpful assistant that can help with code."
    tools: list[Any] = Field(default=DEFAULT_TOOLS)
    silent: bool = False
    response_format: Type[BaseModel] | None = Field(default=None)

    @weave.op
    def step(self, state: AgentState, response_format: BaseModel | None = None) -> AgentState:
        if not self.silent:
            Console.step_start("agent", "green")

        messages = [
            {"role": "system", "content": self.system_message},
        ]
        messages += state.messages

        if self.tools:
            tools = chat_call_tool_params(self.tools)
        else:
            tools = None

        if not self.silent:
            Console.chat_response_start()
        response = litellm.completion(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
            stream=False,
            timeout=60,
            response_format=response_format,
        )

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

        return AgentState(messages=new_history, num_steps=state.num_steps + 1)

    @weave.op
    def run(self, user_prompt: str, max_runtime_seconds: int = -1, max_steps: int = -1):
        if not self.silent:
            Console.welcome(f"Using model: {self.model_name}\nTools: {self.tools}")
        state = AgentState(
            messages=[
                {"role": "user", 
                 "content": user_prompt}]) 

        # Print initial user prompt if available
        if state.messages and state.messages[0]["role"] == "user":
            if not self.silent:
                Console.user_prompt(state.messages[0]["content"])
        
        start_time = time.time()
        
        while True:
            if max_steps > 0 and state.num_steps >= max_steps:
                print(f"Max steps reached: {max_steps}")
                return AgentResponse(final_response=state.messages[-1]["content"], stop_reason="max_steps_reached")
            last_message = state.messages[-1]
            if last_message["role"] == "assistant" and "tool_calls" not in last_message:
                if self.response_format is not None:
                    state = self.step(state, self.response_format)
                    formatted_model = self.response_format.model_validate_json(state.messages[-1]["content"])
                    return AgentResponse(final_response=formatted_model, stop_reason="done")
                return AgentResponse(final_response=last_message["content"], stop_reason="done")
            state = self.step(state)
            if (
                max_runtime_seconds > 0
                and time.time() - start_time > max_runtime_seconds
            ):
                return AgentResponse(final_response=last_message["content"], stop_reason="time_limit_exceeded")