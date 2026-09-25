from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics import AgentGoalAccuracyWithReference


class TestMultiTurn:
    def test_multi_turn(self, llm):
        user_message = HumanMessage(content="What's the weather like in New York City today?")
        ai_initial_response = AIMessage(
            content="Let me check the weather for you.",
            tool_calls=[ToolCall(name="get_weather", args={"location": "New York City"})],
        )
        tool_response = ToolMessage(content="It's sunny and 75 degrees Fahrenheit in New York City today.")
        ai_final_response = AIMessage(content="It's sunny and 75 degrees Fahrenheit in New York City today.")

        conversation = [
            user_message,
            ai_initial_response,
            tool_response,
            ai_final_response,
        ]

        evaluate_llm = LangchainLLMWrapper(llm)
        metric = AgentGoalAccuracyWithReference(llm=evaluate_llm)

        sample = MultiTurnSample(
            user_input=conversation,
            reference="The user is told the current weather in New York City.",
        )

        score = metric.multi_turn_score(sample)
        print("Agent goal accuracy score:", score)
        assert score >= 0.0
