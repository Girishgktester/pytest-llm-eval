from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

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