"""
The agent loop itself -- built by hand (no LangChain/LangGraph) so the
actual mechanics of tool-calling are visible and understood, not hidden
behind a framework.

Uses Groq's free tier via the OpenAI-compatible chat-completions API (no
credit card required -- see console.groq.com). The OpenAI Python SDK works
unchanged against Groq by just pointing base_url at Groq's endpoint, which
is why this uses `openai` as the client library even though there's no
OpenAI account involved.

The core idea, at a glance:
    1. Send the conversation + available tools to the LLM.
    2. If the LLM's response says "I want to call tool X with args Y",
       actually run that Python function, and send the result back as
       part of the conversation.
    3. Repeat until the LLM responds with plain text instead of a tool
       call -- that's the final answer.
That's the entire mechanism behind "agentic AI" tool use. Frameworks like
LangGraph automate and generalize this loop; this is what they're
automating.
"""
import json
import os

from utils.agent_tools import TOOL_FUNCTIONS, to_openai_tool_schemas

MAX_TOOL_ITERATIONS = 8  # raised from 5 after observing broad multi-part questions
# (needing 4-5 distinct tools) exhaust a lower cap on
# Groq's free model, which sometimes re-calls a tool
# it already has data from instead of progressing
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile moved to Groq's Enterprise tier as of late 2026 -- this is the current free-tier tool-calling model

SYSTEM_PROMPT = """You are a research assistant for SmartInvestAI, a stock trend prediction tool.

SCOPE: You ONLY discuss stock/ticker analysis using your tools -- predictions, direction,
sentiment, backtested performance, and explanations for the currently loaded ticker or any
ticker the user names. You do NOT answer unrelated questions (recipes, general trivia, coding
help, or anything else outside this tool's purpose), even if you know the answer. If asked
something off-topic, say briefly that you're scoped to stock analysis for this tool and ask
if they have a ticker-related question instead. Do not lecture or over-explain this -- one
short sentence, then stop.

You have tools that call REAL, backtested ML models -- not general knowledge about stocks.
Ground every claim in actual tool output. Never state a prediction without calling the
relevant tool first.

Tool-calling discipline:
- Before calling a tool, check whether you already called it (visible earlier in this
  conversation) with the same ticker. If so, use that result -- do NOT call it again.
- For a question needing several kinds of information (e.g. "predicted price, direction,
  AND sentiment"), plan to call each DIFFERENT relevant tool once each, then synthesize a
  final answer. Don't call the same tool repeatedly instead of moving to the next one.
- Once you have enough information to answer, respond with your final answer -- don't keep
  calling tools "just in case".

Critical honesty rules, because this project's whole value is being honest about a genuinely
hard, noisy problem:
- ALWAYS mention the backtested baseline comparison (RMSE vs naive baseline, F1 vs naive
  baseline) when discussing a prediction. Never present a number in isolation.
- Match your confidence language to the actual numbers. A 51-53% direction probability is
  close to a coin flip -- say so plainly, don't call it "likely" or "strong". Only use
  confident language (e.g. "clearly", "strongly") when the model meaningfully beats its
  baseline by a wide margin.
- If a model's backtested performance is at or below its naive baseline, say that directly --
  don't bury it or spin it positively.
- NEVER give direct buy/sell/investment advice or tell the user what to do with their money.
  You can describe what the data and models show; the user makes their own decisions.
- If a tool returns an error (e.g. not enough trading history for a newly-listed stock),
  explain that plainly rather than guessing or filling in a number yourself.

Formatting: your responses are rendered as Markdown. When using a table, put any heading or
title on its OWN line, followed by a blank line, THEN the table -- never run a bold title
directly into the same line as a table's header row, since that breaks table parsing. Keep
tables simple (short cell values, no nested formatting inside cells).

Keep responses concise and grounded. You are not a hype machine -- you are the honest,
skeptical research assistant this project was built to be.
"""


def _get_client():
    """Lazy import + init so tests can run without the openai package
    needing a real API key configured."""
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


def run_agent(user_message: str, conversation_history: list = None, model: str = GROQ_MODEL) -> dict:
    """
    Runs one full agent turn: takes the user's message (plus any prior
    conversation), lets the model call tools as needed, and returns the
    final answer along with a trace of which tools were called -- the
    trace is what the frontend shows so the user can see the agent's
    work, not just trust a black-box answer.

    Returns: {"response": str, "tool_calls": list[dict], "messages": list}
    The returned "messages" should be passed back in as conversation_history
    on the next call, so multi-turn context is preserved.
    """
    client = _get_client()
    tool_schemas = to_openai_tool_schemas()

    messages = list(conversation_history) if conversation_history else [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.append({"role": "user", "content": user_message})

    tool_call_trace = []
    # Caches (tool_name, sorted args) -> result FOR THIS TURN ONLY. If the
    # model calls the exact same tool with the exact same arguments again --
    # the failure mode actually observed with Groq's free model on broad
    # multi-part questions -- we don't burn another real tool execution or
    # let it eat another iteration silently; we hand back the same result
    # plus an explicit note telling the model to stop repeating and move on.
    seen_calls = {}

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # Final answer -- no more tools requested, we're done
            messages.append({"role": "assistant", "content": message.content})
            return {"response": message.content or "", "tool_calls": tool_call_trace, "messages": messages}

        # The model wants to call one or more tools. Append the assistant's
        # tool-call request, run each tool, and append the results.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            tool_fn = TOOL_FUNCTIONS.get(tc.function.name)
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                # Open-weight models occasionally emit malformed tool-call
                # arguments -- this is a known Groq/Llama quirk, not
                # something our own code can prevent. Fail this one tool
                # call gracefully instead of crashing the whole turn.
                result = {"error": f"Malformed arguments from model: {tc.function.arguments!r}"}
                args = None
                call_key = None
            else:
                call_key = (tc.function.name, json.dumps(args, sort_keys=True))

            if call_key is not None and call_key in seen_calls:
                # Redundant call -- reuse the cached result (no wasted
                # execution) and explicitly tell the model it already has
                # this, so it doesn't just loop again next iteration.
                result = {
                    **seen_calls[call_key],
                    "_note": "You already called this exact tool with these exact arguments earlier in "
                             "this turn -- this is the same result. Use it, or call a DIFFERENT tool.",
                }
            elif args is not None:
                if tool_fn is None:
                    result = {"error": f"Unknown tool: {tc.function.name}"}
                else:
                    result = tool_fn(**args)
                seen_calls[call_key] = result

            tool_call_trace.append({"tool": tc.function.name, "input": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # Hit the iteration cap -- return whatever we have rather than looping forever
    return {
        "response": "I wasn't able to finish gathering the data needed to answer that -- try a more specific question.",
        "tool_calls": tool_call_trace,
        "messages": messages,
    }