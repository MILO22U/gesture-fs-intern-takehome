"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
"""

def ask_question(vector_store, llm, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer."""
    
    # Check for empty question
    if not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": []
        }
    
    # Retrieve top 3 relevant document chunks
    documentChunks = vector_store.similarity_search(question, k=3)
    
    # Get text from every document chunk
    sources = []
    for chunk in documentChunks:
        sources.append(chunk.page_content)
    
    # Combine all retrieved text
    context = "\n\n".join(sources)
    
    # Create final prompt
    formattedPrompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    # Generate answer
    result = llm(formattedPrompt)
    answer = result[0]["generated_text"]
    
    return {
        "answer": answer,
        "sources": sources
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    # Command line argument
    parser = argparse.ArgumentParser(description="Document Q&A Pipeline")
    parser.add_argument("--query", type=str, help="Ask a single question")
    args = parser.parse_args()

    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    # Check if data directory exists
    if not os.path.isdir(data_dir): #in case the data directory is not found, print an error message and return
        print("Error: data directory not found.")
        return
    
    # Build knowledge base
    vectorStore = build_knowledge_base(data_dir)
    
    # Load local LLM
    llm = get_llm()
    
    # Single question mode
    if args.query is not None:
        result = ask_question(vectorStore, llm, args.query)

        print("\nRetrieved Sources:")
        for source in result["sources"]:
            print("\n", source)

        print("\nAnswer:")
        print(result["answer"])
        return
    
    # Start question answer loop
    while True:
        question = input("\nEnter your question or type quit to exit: ")

        if question.lower() == "quit":
            print("\nExiting Q&A system.")
            break
        
        if not question.strip():
            print("\nPlease enter a question.")
            continue

        result = ask_question(vectorStore, llm, question)
        
        # Print retrieved sources
        print("\nRetrieved Sources:")
        for source in result["sources"]:
            print("\n", source)

        # Print generated answer
        print("\nAnswer:")
        print(result["answer"])


if __name__ == "__main__":
    main()