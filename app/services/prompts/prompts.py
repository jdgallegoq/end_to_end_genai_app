SYS_PROMPT = """
Act as a helpful assistant and answer questions to the best of your ability.
Do not make up answers.
"""

QA_RAG_PROMPT = """
Use only the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know,
don't try to make up an answer. Keep the answer as concise as possible.

{context}

Question: {question}
"""
