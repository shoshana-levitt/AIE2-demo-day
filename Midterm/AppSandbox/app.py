# AI MAKERSPACE MIDTERM PROJECT: META RAG CHATBOT
# Date: 2024-5-2
# Authors: MikeC

# Basic Imports & Setup
import os
from openai import AsyncOpenAI

# Using Chainlit for our UI
import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers import ChatOpenAI

# Getting the API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# RAG pipeline imports and setup code
from langchain.document_loaders import PyMuPDFLoader
docs = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf").load()

import tiktoken
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

from langchain_openai.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_community.vectorstores import Qdrant

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="MetaFin",
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

from langchain_openai import ChatOpenAI
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.prompts import ChatPromptTemplate
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

Use the provided context to answer the user's query. You are a professional financial expert. You always review the provided financial information.  You provide correct, substantiated answers. You may not answer the user's query unless there is a specific context in the following text. If asked about the Board of Directors, then add Mark Zuckerberg as the "Board Chair".
If you do not know the answer, or cannot answer, please respond with "Insufficient data for further analysis, please try again". >>
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)

# Chainlit App
@cl.on_chat_start
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

@cl.on_message
async def main(message: cl.Message):
    chainlit_question = message.content
    #chainlit_question = "What was the total value of 'Cash and cash equivalents' as of December 31, 2023?"
    response = retrieval_augmented_qa_chain.invoke({"question": chainlit_question})
    chainlit_answer = response["response"].content

    msg = cl.Message(content=chainlit_answer)
    await msg.send()
