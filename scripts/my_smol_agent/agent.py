import time
from pydantic import BaseModel, Field
from typing import Any
import weave
from weave.flow.chat_util import OpenAIStream

from my_smol_agent.console import Console
from my_smol_agent.tools import DEFAULT_TOOLS
from my_smol_agent.tool_calling import chat_call_tool_params, perform_tool_calls
import litellm

class AgentState(BaseModel):
    # The chat message history.
    messages: list[Any] = Field(default_factory=list)

class AgentResponse(BaseModel):
    final_response: str = Field(description="The final response from the agent.")
    stop_reason: str = Field(description="The reason the agent stopped.")

class Agent(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    system_message: str = "You are a helpful assistant that can help with code."
    tools: list[Any] = Field(default=DEFAULT_TOOLS)
    silent: bool = False

    @weave.op()
    def step(self, state: AgentState) -> AgentState:
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
        stream = litellm.completion(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
            stream=True,
            timeout=60,
        )
        wrapped_stream = OpenAIStream(stream)  # type: ignore
        for chunk in wrapped_stream:
            if chunk.choices[0].delta.content:
                if not self.silent:
                    Console.chat_message_content_delta(chunk.choices[0].delta.content)

        response = wrapped_stream.final_response()
        response_message = response.choices[0].message
        if response_message.content:
             if not self.silent:
                Console.chat_response_complete(response_message.content)

        new_messages = []
        # we always store the dict representations of messages in agent state
        # instead of mixing in some pydantic objects.
        new_messages.append(response_message.model_dump(exclude_none=True))
        if response_message.tool_calls:
            new_messages.extend(
                perform_tool_calls(self.tools, response_message.tool_calls, self.silent)
            )

        new_history = state.messages + new_messages

        return AgentState(messages=new_history)

    @weave.op
    def run(self, user_prompt: str, max_runtime_seconds: int = -1):
        if not self.silent:
            Console.welcome(f"Using model: {self.model_name}")
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
            last_message = state.messages[-1]
            if last_message["role"] == "assistant" and "tool_calls" not in last_message:
                return AgentResponse(final_response=last_message["content"], stop_reason="done")
            state = self.step(state)
            if (
                max_runtime_seconds > 0
                and time.time() - start_time > max_runtime_seconds
            ):
                return AgentResponse(final_response=last_message["content"], stop_reason="time_limit_exceeded")
    

