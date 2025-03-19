"""
Enhanced RAG system with:
1. Streaming responses
2. File-based prompt input
3. Default prompt file option
"""
import os
import sys
from pathlib import Path
from openai import OpenAI
import json
import tiktoken
import time


# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Initialize tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

# Create prompts directory if it doesn't exist
PROMPTS_DIR = Path("prompts")
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "prompt.md"
os.makedirs(PROMPTS_DIR, exist_ok=True)


def count_tokens(text):
    """Count the number of tokens in the text using tiktoken."""
    tokens = tokenizer.encode(text)
    return len(tokens)


def chunk_document(content, filename, max_tokens=6000, overlap=200):
    """Split document into overlapping chunks that fit within token limits."""
    tokens = tokenizer.encode(content)
    chunks = []

    # If document is small enough, return as is
    if len(tokens) <= max_tokens:
        return [{
            "filename": filename,
            "content": content
        }]

    # Process in chunks
    for i in range(0, len(tokens), max_tokens - overlap):
        # Get chunk with overlap
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)

        # Add to chunks list
        chunks.append({
            "filename": f"{filename} (part {len(chunks) + 1})",
            "content": chunk_text
        })

    print(f"Split '{filename}' into {len(chunks)} chunks")
    return chunks


def load_documents(directory="documents", embedding_cache_file="embedding_cache.json"):
    """Load all markdown documents from the specified directory."""
    docs = []
    dir_path = Path(directory)
    cache_path = Path(embedding_cache_file)
    cache = {}

    # Load cache if it exists
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
                print(f"Loaded embedding cache with {len(cache)} entries")
        except Exception as e:
            print(f"Error loading cache: {e}")

    for file_path in dir_path.glob("*.md"):
        try:
            file_stat = file_path.stat()
            file_modified = file_stat.st_mtime
            cache_key = f"{file_path.name}_{file_modified}"

            # If file exists in cache and hasn't been modified, skip loading
            if cache_key in cache:
                print(f"Using cached version of {file_path.name}")
                # If the document was previously chunked, add all chunks
                if isinstance(cache[cache_key], list):
                    for item in cache[cache_key]:
                        docs.append(item)
                else:
                    docs.append(cache[cache_key])
                continue

            # Otherwise load and process the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Check token count
                token_count = count_tokens(content)
                print(f"Loaded document: {file_path.name} ({token_count} tokens)")

                # If document is too large, chunk it
                if token_count > 6000:
                    chunks = chunk_document(content, file_path.name)
                    for chunk in chunks:
                        docs.append(chunk)
                    # Cache the chunked document
                    cache[cache_key] = chunks
                else:
                    doc = {
                        "filename": file_path.name,
                        "content": content
                    }
                    docs.append(doc)
                    # Cache the document
                    cache[cache_key] = doc

        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    # Save updated cache
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    print(f"Successfully loaded {len(docs)} documents/chunks")
    return docs


def create_embeddings(documents, embedding_cache_file="embedding_cache.json"):
    """Create embeddings for each document using OpenAI API with caching."""
    embeddings = []
    cache_path = Path(embedding_cache_file)
    cache = {}

    # Load cache if it exists
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")

    print("Creating embeddings for documents...")
    for doc in documents:
        try:
            # Create a unique key for this document content
            content_key = f"emb_{hash(doc['content'])}"

            # Check if we already have an embedding for this content
            if content_key in cache and "embedding" in cache[content_key]:
                print(f"Using cached embedding for {doc['filename']}")
                embeddings.append({
                    "filename": doc['filename'],
                    "content": doc['content'],
                    "embedding": cache[content_key]["embedding"]
                })
                continue

            # Create embedding for the document content
            response = client.embeddings.create(
                input=doc["content"],
                model="text-embedding-3-large"
            )

            # Add a slight delay to avoid rate limits
            time.sleep(0.5)

            # Store document with its embedding
            embedding_data = {
                "filename": doc["filename"],
                "content": doc["content"],
                "embedding": response.data[0].embedding
            }
            embeddings.append(embedding_data)

            # Cache the embedding
            cache[content_key] = {"embedding": response.data[0].embedding}

            print(f"Created embedding for {doc['filename']}")

        except Exception as e:
            print(f"Failed to create embedding for {doc['filename']}: {e}")
            # If we fail, we'll try continuing with the documents we have

    # Save updated cache
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    return embeddings


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(y * y for y in b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)


def search_documents(query, document_embeddings, top_k=3):
    """Search for documents relevant to the query."""
    # Create embedding for the query
    query_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    )
    query_embedding = query_response.data[0].embedding

    # Calculate similarity with each document
    similarities = []
    for doc in document_embeddings:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        similarities.append({
            "filename": doc["filename"],
            "content": doc["content"],
            "similarity": similarity
        })

    # Sort by similarity and get top_k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


def query_with_context_streaming(query, relevant_docs):
    """Generate a streaming response to the query using the relevant documents as context."""
    # Combine all relevant document content to use as context
    context = "\n\n".join([f"Document: {doc['filename']}\n{doc['content']}" for doc in relevant_docs])

    # Create the prompt with improved instructions
    messages = [
        {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided documents.
        Answer questions using ONLY information found in the provided context documents.
        If the answer cannot be found in the documents, say 'I don't have enough information to answer this question.'
        Do not use any prior knowledge or information not contained in the provided documents.
        When appropriate, cite the relevant document names in your response."""},
        {"role": "user",
         "content": f"Context documents:\n{context}\n\nBased ONLY on the context documents, answer this query: {query}"}
    ]

    # Generate streaming response
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        stream=True,  # Enable streaming
    )

    # Return the stream object for processing in the main function
    return stream


def get_query_from_file(file_path):
    """Read and return query from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            query = f.read()
        print(f"Loaded query from {file_path} ({len(query)} characters)")
        return query
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def create_default_prompt_if_missing():
    """Create a default prompt.md file if it doesn't exist."""
    if not DEFAULT_PROMPT_FILE.exists():
        with open(DEFAULT_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write("""# Example Prompt

This is a default prompt file. You can edit this file to create your own custom prompt.

You can ask questions like:
- What are the key components described in the documents?
- How does the data extraction process work?
- What are the relationships between different systems?

Feel free to replace this content with your own detailed queries or even paste
structured content like XML or JSON that you want to analyze.
""")
        print(f"Created default prompt file at {DEFAULT_PROMPT_FILE}")


def main():
    """Main function to run the RAG system with streaming responses and file-based prompts."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Welcome to the enhanced RAG system with streaming responses!")

    # Create the default prompt file if it doesn't exist
    create_default_prompt_if_missing()

    # Load documents with caching
    documents = load_documents()
    if not documents:
        print("No documents found. Please add markdown files to the 'documents' directory.")
        sys.exit(1)

    # Create embeddings with caching
    document_embeddings = create_embeddings(documents)
    print(f"Prepared {len(document_embeddings)} document embeddings")

    # Query loop
    print("\nRAG Query System Ready!")
    print("Type 'exit' to quit.")
    print("Type 'file:{filename}' to load a query from a file.")
    print("Press Enter to use the default prompt file.")

    while True:
        user_input = input("\nEnter your query: ")

        # Handle exit command
        if user_input.lower() in ("exit", "quit", "q"):
            break

        # Determine the query source
        if user_input == "":
            # Use default prompt file
            query = get_query_from_file(DEFAULT_PROMPT_FILE)
            if query is None:
                continue
        elif user_input.lower().startswith("file:"):
            # Extract filename and load query from file
            filename = user_input[5:].strip()
            file_path = Path(filename)
            if not file_path.is_absolute():
                # Check if file is in prompts directory first
                prompt_file = PROMPTS_DIR / filename
                if prompt_file.exists():
                    file_path = prompt_file

            query = get_query_from_file(file_path)
            if query is None:
                continue
        else:
            # Use the input directly as the query
            query = user_input

        # Find relevant documents
        print("Searching for relevant documents...")
        relevant_docs = search_documents(query, document_embeddings, top_k=4)

        # Print found documents
        print(f"\nFound {len(relevant_docs)} relevant documents:")
        for i, doc in enumerate(relevant_docs):
            print(f"{i + 1}. {doc['filename']} (similarity: {doc['similarity']:.4f})")

        # Generate streaming response with context
        print("\nResponse:")
        stream = query_with_context_streaming(query, relevant_docs)

        # Process and display the streaming response
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        # Print a newline at the end
        print()


if __name__ == "__main__":
    main()
