# 
# AI MAKERSPACE MIDTERM PROJECT: META RAG CHATBOT
# Date: 2024-5-2
# Authors: MikeC

# ========================================================
# This is the main file for the ChatGPT app
# ========================================================
# Task 1: Basic Imports, Keys and Setup
# Task 2: Component Initalization
#           a) LLM: (OpenAI) Mistral 7B
#           b) Embedding Model: OpenAI text-embedding-3-small
#           c) UI: Chainlit
#           d) Chatbot: ChatOpenAI
# Task 3: RAG Pipeline
# Task 4: Chatbot

# [Task 1] Basic Imports, Keys and Setup
import os

# OpenAI Chat completion, Do we need this?
from openai import AsyncOpenAI  # importing openai for API usage

# Using Chainlit for our UI, future port to Streamlit
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools

# Getting the API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# Task 2: Component Initalization





# ChatOpenAI Templates
system_template = """You are a helpful assistant who always speaks in a pleasant tone!
"""

user_template = """{input}
Think through your response step by step.
"""


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")



    # Question and Answer Chatbot

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
