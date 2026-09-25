from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness

from conftest import REFERENCE, ask_rag_app


class TestRagas:
    def test_ragas_mcp(self, rag_app, llm):
        question = "What is an MCP server?"
        answer = ask_rag_app(rag_app, question)
        print("MCP answer:", answer)

        evaluate_llm = LangchainLLMWrapper(llm)
        metric = FactualCorrectness(llm=evaluate_llm)

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=REFERENCE,
        )

        score = metric.single_turn_score(sample)
        print("Ragas score:", score)
        assert score >= 0.2
