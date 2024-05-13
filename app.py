# AI MAKERSPACE PREPR
# Date: 2024-5-16

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
# Get the DeveloperWeek PDF file (future implementation: direct download from URL)
from langchain.document_loaders import PyMuPDFLoader

# Adjust the URL to the direct download format
file_id = "1JeA-w4kvbI3GHk9Dh_j19_Q0JUDE7hse"
direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Now load the document using the direct URL
docs = PyMuPDFLoader(direct_url).load()

import tiktoken
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

# Split the document into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,           # 500 tokens per chunk, experiment with this value
    chunk_overlap = 50,        # 50 tokens overlap between chunks, experiment with this value
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

# Load the embeddings model
from langchain_openai.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the vector store and retriever from Qdrant
from langchain_community.vectorstores import Qdrant

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="Prepr",
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

Before proceeding to answer about which conference sessions the user should attend, be sure to ask them what key topics they are hoping to learn from the conference, and if there are any specific sessions they are keen on attending. Use the provided context to answer the user's query. You are a professional personal assistant for an executive professional in a high tech company. You help them plan for events and meetings.
You always review the provided event information. You can look up dates and location where event sessions take place from the document. If you do not know the answer, or cannot answer, please respond with "Insufficient data for further analysis, please try again". For each session you suggest, include bullet points with the session title, speaker, company, topic, AI industry relevance, details of their work in AI, main point likely to be made, and three questions to ask the speaker. You end your successful responses with "Is there anything else that I can help you with?". If the user says NO, or any other negative response, then you ask "How did I do?" >>
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
    response = retrieval_augmented_qa_chain.invoke({"question": chainlit_question})
    chainlit_answer = response["response"].content
    msg = cl.Message(content=chainlit_answer)
    await msg.send()
