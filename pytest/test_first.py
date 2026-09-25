import os

import deepeval
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:8b",
    temperature=0.5,
    num_predict=2500,
    reasoning=False,
)

response = llm.invoke("What is the capital of France?")

api_key = os.getenv("CONFIDENT_API_KEY")

if api_key:
    deepeval.login(api_key=api_key)
    print("Logged in to Confident AI")
else:
    print("CONFIDENT_API_KEY not found")


def test_first():
    answer_relevancy_metric = AnswerRelevancyMetric()

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output=response.content,
        expected_output="Paris",
    )

    assert_test(test_case, [answer_relevancy_metric])
