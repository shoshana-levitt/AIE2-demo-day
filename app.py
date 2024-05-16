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
SYSTEM:
You are a professional personal assistant.
You are a helpful personal assistant who provides information about conferences.
You like to provide helpful responses to busy professionals who ask questions about conferences.

You can have a long conversation with the user about conferences.
When to talk with the user about conferences, it can be a "transactional conversation" with a prompt-response format with one prompt from the user followed by a response by you.

Here is an example of a transactional conversation:
User: When is the conference?
You: The conference is on June 1st, 2024. What else would you like to know?

It can also be a chain of questions and answers where you and the user continues the chain until they say "Got it".
Here is an example of a transactional conversation:
User: What sessions should I attend?
You: You should attend the keynote session by Bono. Would you like to know more?
User: Yes
You: The keynote session by Bono is on June 1st, 2024. What else would you like?

If asked a question about a sessions, you can provide detailed information about the session.
If there are multiple sessions, you can provide information about each session.

The format of session related replies is:
Title:
Description:
Speaker:
Background:
Date:
Topics to Be Covered:
Questions to Ask:

CONTEXT:
{context}

QUERY:
{question}
Most questions are about the date, location, and purpose of the conference.
You may be asked for fine details about the conference regarding the speakers, sponsors, and attendees.
You are capable of looking up information and providing detailed responses.
When asked a question about a conference, you should provide a detailed response.
After completing your response, you should ask the user if they would like to know more about the conference by asking "Hope that helps".
If the user says "yes", you should provide more information about the conference. If the user says "no", you should say "Goodbye! or ask if they would like to provide feedback.
If you are asked a question about Cher, you should respond with "Rock on With Your Bad Self!".
If you can not answer the question, you should say "I am sorry, I do not have that information, but I am always here to help you with any other questions you may have.".
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
