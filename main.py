#!/usr/bin/env python3
"""
Iterative Deep Research Script

This script demonstrates how to perform multiple search‐enabled interactions
with OpenAI’s models where each round proposes a follow‑up search until the
accumulated research tokens reach a predefined target (here, 80% of 1M tokens).
It then uses hierarchical reduction to collapse the corpus into ≤32k tokens and
generates a final executive report.
"""

import os, time, textwrap
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken  # For token counting; pip install tiktoken

# ------------------- Load API key and initialize client -------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please define OPENAI_API_KEY in your .env file.")
client = OpenAI(api_key=openai_api_key)

# ------------------- Configuration -------------------
SEARCH_MODEL          = "gpt-4o-search-preview"  # Search‑enabled model
SYNTHESIS_MODEL       = "gpt-4.1-nano"           # Final synthesis model (32k window)
TARGET_CONTEXT_TOKENS = 1_000_000                # Theoretical token budget
STOP_AT_RATIO         = 0.8                      # Stop when ~80% is reached
FINAL_CONTEXT_MAX     = 32_000                   # Final context must be ≤32k tokens

INITIAL_QUESTION = "How have AI‑powered face‑rating apps grown in India over the last 3 years?"

# Use tiktoken to estimate token counts (choose encoding for a relevant model)
token_encoder = tiktoken.encoding_for_model("gpt-4o-mini")

def n_tokens(text: str) -> int:
    """Return an estimated token count for a given text."""
    return len(token_encoder.encode(text))

# ------------------- Function Definitions -------------------
def call_search_model(prompt: str) -> Dict:
    """
    Calls the search‑enabled model with the given prompt.
    Returns a dictionary with the assistant's text and a list of citations.
    """
    response = client.responses.create(
        model=SEARCH_MODEL,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": "medium"
        }],
        input=prompt,
        max_tokens=2048,
        temperature=0.3,
    )
    # Extract the assistant’s message block from the response
    message_item = next(item for item in response if item["type"] == "message")
    text_content = message_item["content"][0]["text"]
    citations = [
        a for a in message_item["content"][0]["annotations"]
        if a["type"] == "url_citation"
    ]
    return {"text": text_content, "citations": citations}

def propose_next_query(history_snippets: List[str]) -> str:
    """
    Proposes a follow‑up search query given the latest research snippets.
    The prompt asks for a single concise query or "STOP" if no further search is needed.
    """
    prompt = f"""
You are building a research dossier on the topic: "{INITIAL_QUESTION}".

Below are recent research notes (each ≤500 words):

{"\n\n---\n\n".join(history_snippets[-3:])}

Based on the above, propose ONE concise follow‑up web search query (with no commentary)
that would fill an important knowledge gap. If no further search is needed, reply "STOP".
"""
    response = client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def hierarchical_reduce(chunks: List[str]) -> str:
    """
    Iteratively reduce a list of research text chunks by summarizing in batches until the
    combined text fits within FINAL_CONTEXT_MAX tokens.
    """
    combined = "\n\n".join(chunks)
    while n_tokens(combined) > FINAL_CONTEXT_MAX:
        new_chunks = []
        # Process in groups of 4 chunks at a time (adjust batch size if desired)
        for i in range(0, len(chunks), 4):
            batch = "\n\n".join(chunks[i:i+4])
            summary_response = client.chat.completions.create(
                model=SYNTHESIS_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Summarize the following research notes in ≤600 words, retaining all citations:\n\n{batch}"
                }],
                max_tokens=2048,
                temperature=0.25,
            )
            summary = summary_response.choices[0].message.content
            new_chunks.append(summary)
            time.sleep(0.5)  # To be gentle with the rate limits
        chunks = new_chunks
        combined = "\n\n".join(chunks)
    return combined

def main():
    corpus: List[str] = []       # Accumulates all research texts
    total_tokens = 0             # Running token count

    # -------- Initial Search Round --------
    print("Starting initial search round...")
    initial_result = call_search_model(INITIAL_QUESTION)
    corpus.append(initial_result["text"])
    total_tokens += n_tokens(initial_result["text"])
    print(f"Round 1: Added {n_tokens(initial_result['text'])} tokens (total: {total_tokens})")
    
    # -------- Iterative Follow‑up Rounds --------
    round_number = 2
    while total_tokens < TARGET_CONTEXT_TOKENS * STOP_AT_RATIO:
        next_query = propose_next_query(corpus)
        if next_query.strip().upper() == "STOP":
            print("Model signalled STOP. No further search needed.")
            break

        result = call_search_model(next_query)
        followup_text = f"### Follow‑up Query: {next_query}\n\n{result['text']}"
        corpus.append(followup_text)
        added_tokens = n_tokens(result["text"])
        total_tokens += added_tokens
        print(f"Round {round_number}: Added {added_tokens} tokens (total: {total_tokens})")
        round_number += 1
        time.sleep(1.0)  # Pause between rounds

    # -------- Collapse Research Corpus to ≤32k Tokens --------
    print("\nReducing research corpus to meet the synthesis window limit (≤32k tokens)...")
    reduced_notes = hierarchical_reduce(corpus)
    print(f"Reduced corpus token count: {n_tokens(reduced_notes)} tokens")

    # -------- Final Synthesis using gpt-4.1-nano --------
    final_prompt = f"""
You are an expert market analyst preparing a comprehensive report.

Topic: {INITIAL_QUESTION}

Using the research notes below (with inline citations), produce:
• An Executive Summary (≈300 words)
• A bulleted list of Key Findings (keeping citation numbers as is)
• A brief 12‑month Market Outlook (≤200 words)

### RESEARCH NOTES
{reduced_notes}
"""
    final_response = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=4096,
        temperature=0.25,
    )
    final_answer = final_response.choices[0].message.content

    print("\nFINAL REPORT:\n")
    print(textwrap.dedent(final_answer))

if __name__ == "__main__":
    main()
