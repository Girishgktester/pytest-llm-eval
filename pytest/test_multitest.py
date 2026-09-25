from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness

from conftest import REFERENCE, get_rag_contexts


class TestMultitest:
    def test_verify_multi_turn(self, llm, rag_retriever):
        test_case = [{
            "user_input": "What is an MCP server?",
            "response": "An MCP server is a lightweight program that exposes tools, resources, and prompts.",
            "reference": REFERENCE,
            "retrieved_contexts": get_rag_contexts(rag_retriever, "What is an MCP server?"),
        }]

        evaluate_llm = LangchainLLMWrapper(llm)
        evaluate_dataset = EvaluationDataset.from_list(test_case)
        results = evaluate(
            dataset=evaluate_dataset,
            metrics=[FactualCorrectness()],
            llm=evaluate_llm,
        )
        # print("Results:----->", results.scores[0].score)
        print("Results:----->", results.to_pandas())
