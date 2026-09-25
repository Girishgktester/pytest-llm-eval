from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval

from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest import ask_rag_app

class TestAnswerRelevancy: 
    def test_answer_relevancy(self, llm):
        answer_relevancy_metric = AnswerRelevancyMetric()
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output=llm.invoke("What is the capital of France?").content,
            expected_output="Paris",
        )
        assert_test(test_case, [answer_relevancy_metric])
        results = evaluate(
            test_cases=[test_case], metrics=[answer_relevancy_metric]
        )
        print(results)

class TestRAG:
    def test_rag(self, rag_app):
        question = "What is an MCP server?"
        answer = ask_rag_app(rag_app, question)
        print("MCP answer:", answer)

        text = answer.lower()
        assert "mcp" in text
        assert "tool" in text
        assert "i don't know" not in text

        correctness = GEval(
            name="Correctness",
            criteria="Check if the answer correctly explains what an MCP server is.",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=0.5,
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output="An MCP server is a lightweight program that exposes tools, resources, and prompts.",
        )
        assert_test(test_case, [correctness])
        results = evaluate(
            test_cases=[test_case], metrics=[AnswerRelevancyMetric(), correctness]
        )
        print(results)