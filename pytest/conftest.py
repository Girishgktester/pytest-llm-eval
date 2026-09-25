import os

import deepeval
import pytest
from deepeval.config.settings import get_settings, reset_settings
from deepeval.metrics.utils import models as metric_models
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def pytest_addoption(parser):
    # Change mode with: pytest --llm-mode=local   or   pytest --llm-mode=online
    # If you skip the flag, LLM_MODE in .env is used (default: local)
    parser.addoption("--llm-mode", default=os.getenv("LLM_MODE", "local"))


def pytest_configure(config):
    mode = config.getoption("--llm-mode")

    if mode == "online":
        os.environ["USE_OPENAI_MODEL"] = "True"
        os.environ["USE_LOCAL_MODEL"] = "False"
    else:
        os.environ["USE_LOCAL_MODEL"] = "True"
        os.environ["USE_OPENAI_MODEL"] = "False"
        os.environ["LOCAL_MODEL_API_KEY"] = "ollama"
        os.environ["OLLAMA_MODEL_NAME"] = os.getenv("LOCAL_OLLAMA_MODEL_NAME", "qwen3:8b")

    # Tell DeepEval to read the values we just set
    reset_settings(reload_dotenv=False)
    metric_models.SETTINGS = get_settings()

    confident_key = os.getenv("CONFIDENT_API_KEY")
    if confident_key:
        deepeval.login(api_key=confident_key)


@pytest.fixture(scope="session")
def llm(request):
    mode = request.config.getoption("--llm-mode")

    if mode == "online":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            max_tokens=250,
        )

    return ChatOllama(
        base_url=os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434"),
        model=os.getenv("LOCAL_OLLAMA_MODEL_NAME", "qwen3:8b"),
        temperature=0.5,
        num_predict=250,
        reasoning=False,
    )


def read_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print("Number of chunks:", len(chunks))
    return chunks


def make_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings


def save_in_chroma(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_langchain_db_openai",
        collection_name="webpage_rag_openai_test",
    )
    return vector_store


def make_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return retriever


def join_chunks(documents):
    text = "\n\n".join(document.page_content for document in documents)
    return text


def make_answer_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        """Use only the context below to answer the question.
        If the answer is not in the context, say: I don't know based on the documents.

    Context:
    {context}

Question: {question}
Answer:""")

    chain = (
        {
            "context": retriever | join_chunks,
            "question": lambda question: question,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@pytest.fixture(scope="session")
def rag_app(llm):
    documents = read_url("https://www.descope.com/learn/post/mcp")
    chunks = split_into_chunks(documents)
    embeddings = make_embeddings()
    vector_store = save_in_chroma(chunks, embeddings)
    retriever = make_retriever(vector_store)
    chain = make_answer_chain(retriever, llm)
    return chain
