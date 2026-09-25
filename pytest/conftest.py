import os

import deepeval
import pytest
from deepeval.config.settings import get_settings, reset_settings
from deepeval.metrics.utils import models as metric_models
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
