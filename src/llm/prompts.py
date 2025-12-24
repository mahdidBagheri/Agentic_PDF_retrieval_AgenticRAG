def build_rag_prompt(query: str, contexts: list) -> str:
    context_text = "\n\n".join(
        f"[Source: {c}"
        for c in contexts
    )

    prompt = f"""
You are a precise assistant. Use ONLY the information in the context below.
If the answer is not contained in the context, say "I don't know."

Context:
{context_text}

Question:
{query}

Answer in clear paragraphs:
"""
    return prompt.strip()