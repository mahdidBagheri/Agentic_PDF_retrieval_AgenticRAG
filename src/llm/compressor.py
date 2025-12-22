class ContextCompressor:
    def __init__(self, llm):
        self.llm = llm

    def compress(self, query: str, chunk_text: str) -> str:
        prompt = f"""
You are a compression assistant.

Task:
Extract ONLY the information from the text that is directly useful for answering the query.
Preserve all factual details, numbers, conditions, and rules.
DO NOT add new information.
DO NOT answer the question.
DO NOT summarize generally.

Query:
{query}

Text:
{chunk_text}

Compressed version:
""".strip()

        return self.llm.generate(prompt)