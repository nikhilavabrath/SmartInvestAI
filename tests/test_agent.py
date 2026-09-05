"""
Tests for utils/agent.py -- the hand-rolled tool-calling loop. The LLM
client is always mocked (no real Groq API calls / cost in this suite).
"""
import json
from types import SimpleNamespace
from unittest import mock

from utils.agent import run_agent


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_system_prompt_restricts_scope_to_stock_analysis():
    """Regression test for a real bug found in manual testing: with no
    scope restriction, the agent happily answered 'how to make chapathi'
    (a recipe question) using its general knowledge, completely unrelated
    to the tool it's meant to assist with. This just asserts the guardrail
    language is still present -- it can't verify the model actually
    obeys it (that's a live-API behavioral question), but it stops a
    future prompt edit from silently deleting the constraint."""
    from utils.agent import SYSTEM_PROMPT
    assert "off-topic" in SYSTEM_PROMPT.lower() or "scope" in SYSTEM_PROMPT.lower()


def test_agent_answers_directly_with_no_tools_needed():
    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.return_value = _response(content="Hi there.", tool_calls=None)

    with mock.patch("utils.agent._get_client", return_value=fake_client):
        result = run_agent("hello")

    assert result["response"] == "Hi there."
    assert result["tool_calls"] == []


def test_agent_executes_a_single_tool_call_and_returns_final_answer():
    first = _response(tool_calls=[_tool_call("c1", "get_backtested_performance", {"ticker": "AAPL"})])
    second = _response(content="AAPL: F1 0.55 vs baseline 0.70.", tool_calls=None)

    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.side_effect = [first, second]

    with mock.patch("utils.agent._get_client", return_value=fake_client), \
            mock.patch("utils.agent_tools.get_cached_report", return_value={"model": {"F1": 0.55}, "naive_baseline": {"F1": 0.70}}):
        result = run_agent("how is AAPL?")

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "get_backtested_performance"
    assert "F1" in result["response"]


def test_agent_handles_malformed_tool_arguments_without_crashing():
    """Regression test for a real, observed Groq/Llama quirk: the model
    occasionally emits invalid JSON as tool arguments."""
    bad_call = _tool_call("c1", "get_next_day_prediction", {"ticker": "AAPL"})
    bad_call.function.arguments = "NOT VALID JSON{{"
    first = _response(tool_calls=[bad_call])
    second = _response(content="I had trouble with that, here's what I know.", tool_calls=None)

    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.side_effect = [first, second]

    with mock.patch("utils.agent._get_client", return_value=fake_client):
        result = run_agent("predict AAPL")

    assert "error" in result["tool_calls"][0]["result"]
    assert result["response"]  # still produced a final answer, didn't crash


def test_agent_recovers_from_redundant_repeated_tool_calls():
    """Regression test for the exact failure observed in manual testing:
    given a broad, multi-part question, Groq's free model sometimes calls
    the SAME tool with the SAME arguments repeatedly instead of progressing
    to the other tools it needs -- exhausting the iteration budget without
    ever answering. The fix: cache same-turn results and inject a
    corrective note, so the model gets nudged back on track within budget."""
    responses = [_response(tool_calls=[_tool_call(f"c{i}", "get_backtested_performance", {"ticker": "YESBANK.NS"})])
                 for i in range(4)]
    responses.append(_response(content="Recovered with a final answer.", tool_calls=None))

    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.side_effect = responses

    with mock.patch("utils.agent._get_client", return_value=fake_client), \
            mock.patch("utils.agent_tools.get_cached_report", return_value={"model": {"F1": 0.4}, "naive_baseline": {"F1": 0.0}}):
        result = run_agent("analyze YESBANK.NS fully")

    assert result["response"] == "Recovered with a final answer."
    # every repeat after the first should carry the corrective note
    assert all("_note" in tc["result"] for tc in result["tool_calls"][1:])


def test_agent_gives_up_gracefully_when_iteration_cap_is_truly_exhausted():
    """If the model keeps calling genuinely DIFFERENT tools (not caught by
    the redundant-call guard) past the cap, the loop must still terminate
    with a clean message -- never an infinite loop or a crash."""
    from utils.agent import MAX_TOOL_ITERATIONS

    responses = [
        _response(tool_calls=[_tool_call(f"c{i}", "get_backtested_performance", {"ticker": f"T{i}"})])
        for i in range(MAX_TOOL_ITERATIONS + 2)
    ]

    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.side_effect = responses

    with mock.patch("utils.agent._get_client", return_value=fake_client), \
            mock.patch("utils.agent_tools.get_cached_report", return_value={"model": {"F1": 0.4}, "naive_baseline": {"F1": 0.0}}):
        result = run_agent("analyze many tickers")

    assert "wasn't able to finish" in result["response"]
    assert fake_client.chat.completions.create.call_count == MAX_TOOL_ITERATIONS