# iterative_deep_research.py
"""
End‑to‑end iterative web‑research pipeline.

• Round‑trips with a search‑enabled model to gather fresh info
• Lets the model propose follow‑up searches
• Stops when accumulated tokens ≈ 80 % of TARGET_CONTEXT
• Reduces everything into <=32 k tokens for a final gpt‑4.1‑nano synthesis
"""

import os, json, time, textwrap
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken                         # pip install tiktoken

load_dotenv()
client = OpenAI()

# ------------------------- configurable knobs -------------------------
SEARCH_MODEL          = "gpt-4o-search-preview"
SYNTHESIS_MODEL       = "gpt-4.1-nano"
TARGET_CONTEXT_TOKENS = 1_000_000       # your theoretical “1 M budget”
STOP_AT_RATIO         = 0.8             # 80 %
FINAL_CONTEXT_MAX     = 32_000          # fits nano today
TOK_EST               = tiktoken.encoding_for_model("gpt-4o-mini")

INITIAL_QUESTION = "How have AI‑powered face‑rating apps grown in India over the last 3 years?"
# ----------------------------------------------------------------------

def n_tokens(text: str) -> int:
    return len(TOK_EST.encode(text))

def call_search_model(prompt: str) -> Dict:
    """Run one search‑enabled call and return the assistant’s message text + citations."""
    resp = client.responses.create(
        model=SEARCH_MODEL,
        tools=[{"type": "web_search_preview", "search_context_size": "medium"}],
        input=prompt,
        max_tokens=2048,
        temperature=0.3,
    )
    msg_item = next(i for i in resp if i["type"] == "message")
    text = msg_item["content"][0]["text"]
    citations = [
        a for a in msg_item["content"][0]["annotations"]
        if a["type"] == "url_citation"
    ]
    return {"text": text, "citations": citations}

def propose_next_query(history_snippets: List[str]) -> str:
    """Ask the search model what to look for next, given what we have so far."""
    prompt = f"""
You are building a research dossier about **{INITIAL_QUESTION}**.

Here is a running log of what we already know (each chunk ≤500 words):

{"\n\n---\n\n".join(history_snippets[-3:])}

Propose ONE concise follow‑up web search query (no commentary) that would
fill an important knowledge gap. If no further search is needed, reply "STOP".
"""
    resp = client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def hierarchical_reduce(chunks: List[str]) -> str:
    """Fold an arbitrarily long list of chunks down to <=FINAL_CONTEXT_MAX tokens."""
    while n_tokens("\n\n".join(chunks)) > FINAL_CONTEXT_MAX:
        new_chunks = []
        for i in range(0, len(chunks), 4):               # map‑reduce 4→1
            batch = "\n\n".join(chunks[i:i+4])
            summary = client.chat.completions.create(
                model=SYNTHESIS_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Summarise the following research notes in ≤600 words, keep citations:\n{batch}"
                }],
                max_tokens=2048,
                temperature=0.25,
            ).choices[0].message.content
            new_chunks.append(summary)
            time.sleep(0.5)  # stay inside rate limits
        chunks = new_chunks
    return "\n\n".join(chunks)

def main():
    corpus: List[str] = []
    total_tokens = 0

    # -------- initial search round --------
    result = call_search_model(INITIAL_QUESTION)
    corpus.append(result["text"])
    total_tokens += n_tokens(result["text"])
    print(f"Round 1  | +{n_tokens(result['text'])} tok  | total {total_tokens}")

    # -------- iterative follow‑ups --------
    round_n = 2
    while total_tokens < TARGET_CONTEXT_TOKENS * STOP_AT_RATIO:
        next_q = propose_next_query(corpus)
        if next_q.upper().startswith("STOP"):
            print("Model signalled STOP.")
            break

        result = call_search_model(next_q)
        corpus.append(f"### Follow‑up query: {next_q}\n\n{result['text']}")
        added = n_tokens(result["text"])
        total_tokens += added
        print(f"Round {round_n:<2}| +{added} tok | total {total_tokens}")
        round_n += 1
        time.sleep(1.0)  # be gentle with the API

    # -------- collapse to ≤32 k for nano --------
    print("\nReducing to <=32 k tokens for final synthesis…")
    reduced_notes = hierarchical_reduce(corpus)

    final_prompt = f"""
You are writing a comprehensive market report.

Topic: {INITIAL_QUESTION}

Below is all the research we collected (with citations). Write:
• An executive summary (≈300 words)
• Key findings (bullets, keep citations)
• 12‑month outlook (≤200 words)

### RESEARCH NOTES
{reduced_notes}
"""
    final = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=4096,
        temperature=0.25,
    ).choices[0].message.content

    print("\n" + textwrap.dedent(final))

if __name__ == "__main__":
    main()
