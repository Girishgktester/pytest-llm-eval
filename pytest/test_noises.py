from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import NoiseSensitivity

from conftest import REFERENCE, ask_rag_app, get_rag_contexts


class TestNoises:
    def test_noise_sensitivity(self, rag_app, rag_retriever, llm):
        question = "What is an MCP server?"
        answer = ask_rag_app(rag_app, question)
        contexts = get_rag_contexts(rag_retriever, question)

        evaluate_llm = LangchainLLMWrapper(llm)
        metric = NoiseSensitivity(llm=evaluate_llm)

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=REFERENCE,
            retrieved_contexts=contexts,
        )

        score = metric.single_turn_score(sample)
        print("Noise sensitivity score:", score)
        assert score >= 0.0
