# Extraction Strategies and Capabilities from the `crawl4ai` Library

Here are the documentation files from the Python package `crawl4ai`

Carefully reference these so as to fully leverage the package when considering solutions for extraction.

# /code/open-source/crawl4ai/crawl4ai/docs/md_v2/extraction/chunking.md

```plaintext
# Chunking Strategies
Chunking strategies are critical for dividing large texts into manageable parts, enabling effective content processing and extraction. These strategies are foundational in cosine similarity-based extraction techniques, which allow users to retrieve only the most relevant chunks of content for a given query. Additionally, they facilitate direct integration into RAG (Retrieval-Augmented Generation) systems for structured and scalable workflows.

### Why Use Chunking?
1. **Cosine Similarity and Query Relevance**: Prepares chunks for semantic similarity analysis.
2. **RAG System Integration**: Seamlessly processes and stores chunks for retrieval.
3. **Structured Processing**: Allows for diverse segmentation methods, such as sentence-based, topic-based, or windowed approaches.

### Methods of Chunking

#### 1. Regex-Based Chunking
Splits text based on regular expression patterns, useful for coarse segmentation.

**Code Example**:
```python
class RegexChunking:
    def __init__(self, patterns=None):
        self.patterns = patterns or [r'\n\n']  # Default pattern for paragraphs

    def chunk(self, text):
        paragraphs = [text]
        for pattern in self.patterns:
            paragraphs = [seg for p in paragraphs for seg in re.split(pattern, p)]
        return paragraphs

# Example Usage
text = """This is the first paragraph.

This is the second paragraph."""
chunker = RegexChunking()
print(chunker.chunk(text))
```

#### 2. Sentence-Based Chunking
Divides text into sentences using NLP tools, ideal for extracting meaningful statements.

**Code Example**:
```python
from nltk.tokenize import sent_tokenize

class NlpSentenceChunking:
    def chunk(self, text):
        sentences = sent_tokenize(text)
        return [sentence.strip() for sentence in sentences]

# Example Usage
text = "This is sentence one. This is sentence two."
chunker = NlpSentenceChunking()
print(chunker.chunk(text))
```

#### 3. Topic-Based Segmentation
Uses algorithms like TextTiling to create topic-coherent chunks.

**Code Example**:
```python
from nltk.tokenize import TextTilingTokenizer

class TopicSegmentationChunking:
    def __init__(self):
        self.tokenizer = TextTilingTokenizer()

    def chunk(self, text):
        return self.tokenizer.tokenize(text)

# Example Usage
text = """This is an introduction.
This is a detailed discussion on the topic."""
chunker = TopicSegmentationChunking()
print(chunker.chunk(text))
```

#### 4. Fixed-Length Word Chunking
Segments text into chunks of a fixed word count.

**Code Example**:
```python
class FixedLengthWordChunking:
    def __init__(self, chunk_size=100):
        self.chunk_size = chunk_size

    def chunk(self, text):
        words = text.split()
        return [' '.join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

# Example Usage
text = "This is a long text with many words to be chunked into fixed sizes."
chunker = FixedLengthWordChunking(chunk_size=5)
print(chunker.chunk(text))
```

#### 5. Sliding Window Chunking
Generates overlapping chunks for better contextual coherence.

**Code Example**:
```python
class SlidingWindowChunking:
    def __init__(self, window_size=100, step=50):
        self.window_size = window_size
        self.step = step

    def chunk(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(' '.join(words[i:i + self.window_size]))
        return chunks

# Example Usage
text = "This is a long text to demonstrate sliding window chunking."
chunker = SlidingWindowChunking(window_size=5, step=2)
print(chunker.chunk(text))
```

### Combining Chunking with Cosine Similarity
To enhance the relevance of extracted content, chunking strategies can be paired with cosine similarity techniques. Here’s an example workflow:

**Code Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarityExtractor:
    def __init__(self, query):
        self.query = query
        self.vectorizer = TfidfVectorizer()

    def find_relevant_chunks(self, chunks):
        vectors = self.vectorizer.fit_transform([self.query] + chunks)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        return [(chunks[i], similarities[i]) for i in range(len(chunks))]

# Example Workflow
text = """This is a sample document. It has multiple sentences. 
We are testing chunking and similarity."""

chunker = SlidingWindowChunking(window_size=5, step=3)
chunks = chunker.chunk(text)
query = "testing chunking"
extractor = CosineSimilarityExtractor(query)
relevant_chunks = extractor.find_relevant_chunks(chunks)

print(relevant_chunks)
```

```
# /code/open-source/crawl4ai/crawl4ai/docs/md_v2/extraction/clustring-strategies.md

```plaintext
# Cosine Strategy

The Cosine Strategy in Crawl4AI uses similarity-based clustering to identify and extract relevant content sections from web pages. This strategy is particularly useful when you need to find and extract content based on semantic similarity rather than structural patterns.

## How It Works

The Cosine Strategy:
1. Breaks down page content into meaningful chunks
2. Converts text into vector representations
3. Calculates similarity between chunks
4. Clusters similar content together
5. Ranks and filters content based on relevance

## Basic Usage

```python
from crawl4ai.extraction_strategy import CosineStrategy

strategy = CosineStrategy(
    semantic_filter="product reviews",    # Target content type
    word_count_threshold=10,             # Minimum words per cluster
    sim_threshold=0.3                    # Similarity threshold
)

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(
        url="https://example.com/reviews",
        extraction_strategy=strategy
    )
    
    content = result.extracted_content
```

## Configuration Options

### Core Parameters

```python
CosineStrategy(
    # Content Filtering
    semantic_filter: str = None,       # Keywords/topic for content filtering
    word_count_threshold: int = 10,    # Minimum words per cluster
    sim_threshold: float = 0.3,        # Similarity threshold (0.0 to 1.0)
    
    # Clustering Parameters
    max_dist: float = 0.2,            # Maximum distance for clustering
    linkage_method: str = 'ward',      # Clustering linkage method
    top_k: int = 3,                   # Number of top categories to extract
    
    # Model Configuration
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',  # Embedding model
    
    verbose: bool = False             # Enable logging
)
```

### Parameter Details

1. **semantic_filter**
   - Sets the target topic or content type
   - Use keywords relevant to your desired content
   - Example: "technical specifications", "user reviews", "pricing information"

2. **sim_threshold**
   - Controls how similar content must be to be grouped together
   - Higher values (e.g., 0.8) mean stricter matching
   - Lower values (e.g., 0.3) allow more variation
   ```python
   # Strict matching
   strategy = CosineStrategy(sim_threshold=0.8)
   
   # Loose matching
   strategy = CosineStrategy(sim_threshold=0.3)
   ```

3. **word_count_threshold**
   - Filters out short content blocks
   - Helps eliminate noise and irrelevant content
   ```python
   # Only consider substantial paragraphs
   strategy = CosineStrategy(word_count_threshold=50)
   ```

4. **top_k**
   - Number of top content clusters to return
   - Higher values return more diverse content
   ```python
   # Get top 5 most relevant content clusters
   strategy = CosineStrategy(top_k=5)
   ```

## Use Cases

### 1. Article Content Extraction
```python
strategy = CosineStrategy(
    semantic_filter="main article content",
    word_count_threshold=100,  # Longer blocks for articles
    top_k=1                   # Usually want single main content
)

result = await crawler.arun(
    url="https://example.com/blog/post",
    extraction_strategy=strategy
)
```

### 2. Product Review Analysis
```python
strategy = CosineStrategy(
    semantic_filter="customer reviews and ratings",
    word_count_threshold=20,   # Reviews can be shorter
    top_k=10,                 # Get multiple reviews
    sim_threshold=0.4         # Allow variety in review content
)
```

### 3. Technical Documentation
```python
strategy = CosineStrategy(
    semantic_filter="technical specifications documentation",
    word_count_threshold=30,
    sim_threshold=0.6,        # Stricter matching for technical content
    max_dist=0.3             # Allow related technical sections
)
```

## Advanced Features

### Custom Clustering
```python
strategy = CosineStrategy(
    linkage_method='complete',  # Alternative clustering method
    max_dist=0.4,              # Larger clusters
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # Multilingual support
)
```

### Content Filtering Pipeline
```python
strategy = CosineStrategy(
    semantic_filter="pricing plans features",
    word_count_threshold=15,
    sim_threshold=0.5,
    top_k=3
)

async def extract_pricing_features(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            extraction_strategy=strategy
        )
        
        if result.success:
            content = json.loads(result.extracted_content)
            return {
                'pricing_features': content,
                'clusters': len(content),
                'similarity_scores': [item['score'] for item in content]
            }
```

## Best Practices

1. **Adjust Thresholds Iteratively**
   - Start with default values
   - Adjust based on results
   - Monitor clustering quality

2. **Choose Appropriate Word Count Thresholds**
   - Higher for articles (100+)
   - Lower for reviews/comments (20+)
   - Medium for product descriptions (50+)

3. **Optimize Performance**
   ```python
   strategy = CosineStrategy(
       word_count_threshold=10,  # Filter early
       top_k=5,                 # Limit results
       verbose=True             # Monitor performance
   )
   ```

4. **Handle Different Content Types**
   ```python
   # For mixed content pages
   strategy = CosineStrategy(
       semantic_filter="product features",
       sim_threshold=0.4,      # More flexible matching
       max_dist=0.3,          # Larger clusters
       top_k=3                # Multiple relevant sections
   )
   ```

## Error Handling

```python
try:
    result = await crawler.arun(
        url="https://example.com",
        extraction_strategy=strategy
    )
    
    if result.success:
        content = json.loads(result.extracted_content)
        if not content:
            print("No relevant content found")
    else:
        print(f"Extraction failed: {result.error_message}")
        
except Exception as e:
    print(f"Error during extraction: {str(e)}")
```

The Cosine Strategy is particularly effective when:
- Content structure is inconsistent
- You need semantic understanding
- You want to find similar content blocks
- Structure-based extraction (CSS/XPath) isn't reliable

It works well with other strategies and can be used as a pre-processing step for LLM-based extraction.
```
# /code/open-source/crawl4ai/crawl4ai/docs/md_v2/extraction/llm-strategies.md

```plaintext
# Extracting JSON (LLM)

In some cases, you need to extract **complex or unstructured** information from a webpage that a simple CSS/XPath schema cannot easily parse. Or you want **AI**-driven insights, classification, or summarization. For these scenarios, Crawl4AI provides an **LLM-based extraction strategy** that:

1. Works with **any** large language model supported by [LightLLM](https://github.com/LightLLM) (Ollama, OpenAI, Claude, and more).  
2. Automatically splits content into chunks (if desired) to handle token limits, then combines results.  
3. Lets you define a **schema** (like a Pydantic model) or a simpler “block” extraction approach.

**Important**: LLM-based extraction can be slower and costlier than schema-based approaches. If your page data is highly structured, consider using [`JsonCssExtractionStrategy`](./no-llm-strategies.md) or [`JsonXPathExtractionStrategy`](./no-llm-strategies.md) first. But if you need AI to interpret or reorganize content, read on!

---

## 1. Why Use an LLM?

- **Complex Reasoning**: If the site’s data is unstructured, scattered, or full of natural language context.  
- **Semantic Extraction**: Summaries, knowledge graphs, or relational data that require comprehension.  
- **Flexible**: You can pass instructions to the model to do more advanced transformations or classification.

---

## 2. Provider-Agnostic via LightLLM

Crawl4AI uses a “provider string” (e.g., `"openai/gpt-4o"`, `"ollama/llama2.0"`, `"aws/titan"`) to identify your LLM. **Any** model that LightLLM supports is fair game. You just provide:

- **`provider`**: The `<provider>/<model_name>` identifier (e.g., `"openai/gpt-4"`, `"ollama/llama2"`, `"huggingface/google-flan"`, etc.).  
- **`api_token`**: If needed (for OpenAI, HuggingFace, etc.); local models or Ollama might not require it.  
- **`api_base`** (optional): If your provider has a custom endpoint.  

This means you **aren’t locked** into a single LLM vendor. Switch or experiment easily.

---

## 3. How LLM Extraction Works

### 3.1 Flow

1. **Chunking** (optional): The HTML or markdown is split into smaller segments if it’s very long (based on `chunk_token_threshold`, overlap, etc.).  
2. **Prompt Construction**: For each chunk, the library forms a prompt that includes your **`instruction`** (and possibly schema or examples).  
3. **LLM Inference**: Each chunk is sent to the model in parallel or sequentially (depending on your concurrency).  
4. **Combining**: The results from each chunk are merged and parsed into JSON.

### 3.2 `extraction_type`

- **`"schema"`**: The model tries to return JSON conforming to your Pydantic-based schema.  
- **`"block"`**: The model returns freeform text, or smaller JSON structures, which the library collects.  

For structured data, `"schema"` is recommended. You provide `schema=YourPydanticModel.model_json_schema()`.

---

## 4. Key Parameters

Below is an overview of important LLM extraction parameters. All are typically set inside `LLMExtractionStrategy(...)`. You then put that strategy in your `CrawlerRunConfig(..., extraction_strategy=...)`.

1. **`provider`** (str): e.g., `"openai/gpt-4"`, `"ollama/llama2"`.  
2. **`api_token`** (str): The API key or token for that model. May not be needed for local models.  
3. **`schema`** (dict): A JSON schema describing the fields you want. Usually generated by `YourModel.model_json_schema()`.  
4. **`extraction_type`** (str): `"schema"` or `"block"`.  
5. **`instruction`** (str): Prompt text telling the LLM what you want extracted. E.g., “Extract these fields as a JSON array.”  
6. **`chunk_token_threshold`** (int): Maximum tokens per chunk. If your content is huge, you can break it up for the LLM.  
7. **`overlap_rate`** (float): Overlap ratio between adjacent chunks. E.g., `0.1` means 10% of each chunk is repeated to preserve context continuity.  
8. **`apply_chunking`** (bool): Set `True` to chunk automatically. If you want a single pass, set `False`.  
9. **`input_format`** (str): Determines **which** crawler result is passed to the LLM. Options include:  
   - `"markdown"`: The raw markdown (default).  
   - `"fit_markdown"`: The filtered “fit” markdown if you used a content filter.  
   - `"html"`: The cleaned or raw HTML.  
10. **`extra_args`** (dict): Additional LLM parameters like `temperature`, `max_tokens`, `top_p`, etc.  
11. **`show_usage()`**: A method you can call to print out usage info (token usage per chunk, total cost if known).  

**Example**:

```python
extraction_strategy = LLMExtractionStrategy(
    llm_config = LLMConfig(provider="openai/gpt-4", api_token="YOUR_OPENAI_KEY"),
    schema=MyModel.model_json_schema(),
    extraction_type="schema",
    instruction="Extract a list of items from the text with 'name' and 'price' fields.",
    chunk_token_threshold=1200,
    overlap_rate=0.1,
    apply_chunking=True,
    input_format="html",
    extra_args={"temperature": 0.1, "max_tokens": 1000},
    verbose=True
)
```

---

## 5. Putting It in `CrawlerRunConfig`

**Important**: In Crawl4AI, all strategy definitions should go inside the `CrawlerRunConfig`, not directly as a param in `arun()`. Here’s a full example:

```python
import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class Product(BaseModel):
    name: str
    price: str

async def main():
    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(provider="openai/gpt-4o-mini", api_token=os.getenv('OPENAI_API_KEY')),
        schema=Product.schema_json(), # Or use model_json_schema()
        extraction_type="schema",
        instruction="Extract all product objects with 'name' and 'price' from the content.",
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",   # or "html", "fit_markdown"
        extra_args={"temperature": 0.0, "max_tokens": 800}
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # 4. Let's say we want to crawl a single page
        result = await crawler.arun(
            url="https://example.com/products",
            config=crawl_config
        )

        if result.success:
            # 5. The extracted content is presumably JSON
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)
            
            # 6. Show usage stats
            llm_strategy.show_usage()  # prints token usage
        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Chunking Details

### 6.1 `chunk_token_threshold`

If your page is large, you might exceed your LLM’s context window. **`chunk_token_threshold`** sets the approximate max tokens per chunk. The library calculates word→token ratio using `word_token_rate` (often ~0.75 by default). If chunking is enabled (`apply_chunking=True`), the text is split into segments.

### 6.2 `overlap_rate`

To keep context continuous across chunks, we can overlap them. E.g., `overlap_rate=0.1` means each subsequent chunk includes 10% of the previous chunk’s text. This is helpful if your needed info might straddle chunk boundaries.

### 6.3 Performance & Parallelism

By chunking, you can potentially process multiple chunks in parallel (depending on your concurrency settings and the LLM provider). This reduces total time if the site is huge or has many sections.

---

## 7. Input Format

By default, **LLMExtractionStrategy** uses `input_format="markdown"`, meaning the **crawler’s final markdown** is fed to the LLM. You can change to:

- **`html`**: The cleaned HTML or raw HTML (depending on your crawler config) goes into the LLM.  
- **`fit_markdown`**: If you used, for instance, `PruningContentFilter`, the “fit” version of the markdown is used. This can drastically reduce tokens if you trust the filter.  
- **`markdown`**: Standard markdown output from the crawler’s `markdown_generator`.

This setting is crucial: if the LLM instructions rely on HTML tags, pick `"html"`. If you prefer a text-based approach, pick `"markdown"`.

```python
LLMExtractionStrategy(
    # ...
    input_format="html",  # Instead of "markdown" or "fit_markdown"
)
```

---

## 8. Token Usage & Show Usage

To keep track of tokens and cost, each chunk is processed with an LLM call. We record usage in:

- **`usages`** (list): token usage per chunk or call.  
- **`total_usage`**: sum of all chunk calls.  
- **`show_usage()`**: prints a usage report (if the provider returns usage data).

```python
llm_strategy = LLMExtractionStrategy(...)
# ...
llm_strategy.show_usage()
# e.g. “Total usage: 1241 tokens across 2 chunk calls”
```

If your model provider doesn’t return usage info, these fields might be partial or empty.

---

## 9. Example: Building a Knowledge Graph

Below is a snippet combining **`LLMExtractionStrategy`** with a Pydantic schema for a knowledge graph. Notice how we pass an **`instruction`** telling the model what to parse.

```python
import os
import json
import asyncio
from typing import List
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class Entity(BaseModel):
    name: str
    description: str

class Relationship(BaseModel):
    entity1: Entity
    entity2: Entity
    description: str
    relation_type: str

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

async def main():
    # LLM extraction strategy
    llm_strat = LLMExtractionStrategy(
        provider="openai/gpt-4",
        api_token=os.getenv('OPENAI_API_KEY'),
        schema=KnowledgeGraph.schema_json(),
        extraction_type="schema",
        instruction="Extract entities and relationships from the content. Return valid JSON.",
        chunk_token_threshold=1400,
        apply_chunking=True,
        input_format="html",
        extra_args={"temperature": 0.1, "max_tokens": 1500}
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strat,
        cache_mode=CacheMode.BYPASS
    )

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        # Example page
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, config=crawl_config)

        if result.success:
            with open("kb_result.json", "w", encoding="utf-8") as f:
                f.write(result.extracted_content)
            llm_strat.show_usage()
        else:
            print("Crawl failed:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())
```

**Key Observations**:

- **`extraction_type="schema"`** ensures we get JSON fitting our `KnowledgeGraph`.  
- **`input_format="html"`** means we feed HTML to the model.  
- **`instruction`** guides the model to output a structured knowledge graph.  

---

## 10. Best Practices & Caveats

1. **Cost & Latency**: LLM calls can be slow or expensive. Consider chunking or smaller coverage if you only need partial data.  
2. **Model Token Limits**: If your page + instruction exceed the context window, chunking is essential.  
3. **Instruction Engineering**: Well-crafted instructions can drastically improve output reliability.  
4. **Schema Strictness**: `"schema"` extraction tries to parse the model output as JSON. If the model returns invalid JSON, partial extraction might happen, or you might get an error.  
5. **Parallel vs. Serial**: The library can process multiple chunks in parallel, but you must watch out for rate limits on certain providers.  
6. **Check Output**: Sometimes, an LLM might omit fields or produce extraneous text. You may want to post-validate with Pydantic or do additional cleanup.

---

## 11. Conclusion

**LLM-based extraction** in Crawl4AI is **provider-agnostic**, letting you choose from hundreds of models via LightLLM. It’s perfect for **semantically complex** tasks or generating advanced structures like knowledge graphs. However, it’s **slower** and potentially costlier than schema-based approaches. Keep these tips in mind:

- Put your LLM strategy **in `CrawlerRunConfig`**.  
- Use **`input_format`** to pick which form (markdown, HTML, fit_markdown) the LLM sees.  
- Tweak **`chunk_token_threshold`**, **`overlap_rate`**, and **`apply_chunking`** to handle large content efficiently.  
- Monitor token usage with `show_usage()`.

If your site’s data is consistent or repetitive, consider [`JsonCssExtractionStrategy`](./no-llm-strategies.md) first for speed and simplicity. But if you need an **AI-driven** approach, `LLMExtractionStrategy` offers a flexible, multi-provider solution for extracting structured JSON from any website.

**Next Steps**:

1. **Experiment with Different Providers**  
   - Try switching the `provider` (e.g., `"ollama/llama2"`, `"openai/gpt-4o"`, etc.) to see differences in speed, accuracy, or cost.  
   - Pass different `extra_args` like `temperature`, `top_p`, and `max_tokens` to fine-tune your results.

2. **Performance Tuning**  
   - If pages are large, tweak `chunk_token_threshold`, `overlap_rate`, or `apply_chunking` to optimize throughput.  
   - Check the usage logs with `show_usage()` to keep an eye on token consumption and identify potential bottlenecks.

3. **Validate Outputs**  
   - If using `extraction_type="schema"`, parse the LLM’s JSON with a Pydantic model for a final validation step.  
   - Log or handle any parse errors gracefully, especially if the model occasionally returns malformed JSON.

4. **Explore Hooks & Automation**  
   - Integrate LLM extraction with [hooks](../advanced/hooks-auth.md) for complex pre/post-processing.  
   - Use a multi-step pipeline: crawl, filter, LLM-extract, then store or index results for further analysis.

**Last Updated**: 2025-01-01

---

That’s it for **Extracting JSON (LLM)**—now you can harness AI to parse, classify, or reorganize data on the web. Happy crawling!
```
# /code/open-source/crawl4ai/crawl4ai/docs/md_v2/extraction/no-llm-strategies.md

```plaintext
# Extracting JSON (No LLM)

One of Crawl4AI’s **most powerful** features is extracting **structured JSON** from websites **without** relying on large language models. By defining a **schema** with CSS or XPath selectors, you can extract data instantly—even from complex or nested HTML structures—without the cost, latency, or environmental impact of an LLM.

**Why avoid LLM for basic extractions?**

1. **Faster & Cheaper**: No API calls or GPU overhead.  
2. **Lower Carbon Footprint**: LLM inference can be energy-intensive. A well-defined schema is practically carbon-free.  
3. **Precise & Repeatable**: CSS/XPath selectors do exactly what you specify. LLM outputs can vary or hallucinate.  
4. **Scales Readily**: For thousands of pages, schema-based extraction runs quickly and in parallel.

Below, we’ll explore how to craft these schemas and use them with **JsonCssExtractionStrategy** (or **JsonXPathExtractionStrategy** if you prefer XPath). We’ll also highlight advanced features like **nested fields** and **base element attributes**.

---

## 1. Intro to Schema-Based Extraction

A schema defines:

1. A **base selector** that identifies each “container” element on the page (e.g., a product row, a blog post card).  
2. **Fields** describing which CSS/XPath selectors to use for each piece of data you want to capture (text, attribute, HTML block, etc.).  
3. **Nested** or **list** types for repeated or hierarchical structures.  

For example, if you have a list of products, each one might have a name, price, reviews, and “related products.” This approach is faster and more reliable than an LLM for consistent, structured pages.

---

## 2. Simple Example: Crypto Prices

Let’s begin with a **simple** schema-based extraction using the `JsonCssExtractionStrategy`. Below is a snippet that extracts cryptocurrency prices from a site (similar to the legacy Coinbase example). Notice we **don’t** call any LLM:

```python
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

async def extract_crypto_prices():
    # 1. Define a simple extraction schema
    schema = {
        "name": "Crypto Prices",
        "baseSelector": "div.crypto-row",    # Repeated elements
        "fields": [
            {
                "name": "coin_name",
                "selector": "h2.coin-name",
                "type": "text"
            },
            {
                "name": "price",
                "selector": "span.coin-price",
                "type": "text"
            }
        ]
    }

    # 2. Create the extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    # 3. Set up your crawler config (if needed)
    config = CrawlerRunConfig(
        # e.g., pass js_code or wait_for if the page is dynamic
        # wait_for="css:.crypto-row:nth-child(20)"
        cache_mode = CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        # 4. Run the crawl and extraction
        result = await crawler.arun(
            url="https://example.com/crypto-prices",
            
            config=config
        )

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        # 5. Parse the extracted JSON
        data = json.loads(result.extracted_content)
        print(f"Extracted {len(data)} coin entries")
        print(json.dumps(data[0], indent=2) if data else "No data found")

asyncio.run(extract_crypto_prices())
```

**Highlights**:

- **`baseSelector`**: Tells us where each “item” (crypto row) is.  
- **`fields`**: Two fields (`coin_name`, `price`) using simple CSS selectors.  
- Each field defines a **`type`** (e.g., `text`, `attribute`, `html`, `regex`, etc.).

No LLM is needed, and the performance is **near-instant** for hundreds or thousands of items.

---

### **XPath Example with `raw://` HTML**

Below is a short example demonstrating **XPath** extraction plus the **`raw://`** scheme. We’ll pass a **dummy HTML** directly (no network request) and define the extraction strategy in `CrawlerRunConfig`.

```python
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonXPathExtractionStrategy

async def extract_crypto_prices_xpath():
    # 1. Minimal dummy HTML with some repeating rows
    dummy_html = """
    <html>
      <body>
        <div class='crypto-row'>
          <h2 class='coin-name'>Bitcoin</h2>
          <span class='coin-price'>$28,000</span>
        </div>
        <div class='crypto-row'>
          <h2 class='coin-name'>Ethereum</h2>
          <span class='coin-price'>$1,800</span>
        </div>
      </body>
    </html>
    """

    # 2. Define the JSON schema (XPath version)
    schema = {
        "name": "Crypto Prices via XPath",
        "baseSelector": "//div[@class='crypto-row']",
        "fields": [
            {
                "name": "coin_name",
                "selector": ".//h2[@class='coin-name']",
                "type": "text"
            },
            {
                "name": "price",
                "selector": ".//span[@class='coin-price']",
                "type": "text"
            }
        ]
    }

    # 3. Place the strategy in the CrawlerRunConfig
    config = CrawlerRunConfig(
        extraction_strategy=JsonXPathExtractionStrategy(schema, verbose=True)
    )

    # 4. Use raw:// scheme to pass dummy_html directly
    raw_url = f"raw://{dummy_html}"

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=raw_url,
            config=config
        )

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        data = json.loads(result.extracted_content)
        print(f"Extracted {len(data)} coin rows")
        if data:
            print("First item:", data[0])

asyncio.run(extract_crypto_prices_xpath())
```

**Key Points**:

1. **`JsonXPathExtractionStrategy`** is used instead of `JsonCssExtractionStrategy`.  
2. **`baseSelector`** and each field’s `"selector"` use **XPath** instead of CSS.  
3. **`raw://`** lets us pass `dummy_html` with no real network request—handy for local testing.  
4. Everything (including the extraction strategy) is in **`CrawlerRunConfig`**.  

That’s how you keep the config self-contained, illustrate **XPath** usage, and demonstrate the **raw** scheme for direct HTML input—all while avoiding the old approach of passing `extraction_strategy` directly to `arun()`.

---

## 3. Advanced Schema & Nested Structures

Real sites often have **nested** or repeated data—like categories containing products, which themselves have a list of reviews or features. For that, we can define **nested** or **list** (and even **nested_list**) fields.

### Sample E-Commerce HTML

We have a **sample e-commerce** HTML file on GitHub (example):
```
https://gist.githubusercontent.com/githubusercontent/2d7b8ba3cd8ab6cf3c8da771ddb36878/raw/1ae2f90c6861ce7dd84cc50d3df9920dee5e1fd2/sample_ecommerce.html
```
This snippet includes categories, products, features, reviews, and related items. Let’s see how to define a schema that fully captures that structure **without LLM**.

```python
schema = {
    "name": "E-commerce Product Catalog",
    "baseSelector": "div.category",
    # (1) We can define optional baseFields if we want to extract attributes 
    # from the category container
    "baseFields": [
        {"name": "data_cat_id", "type": "attribute", "attribute": "data-cat-id"}, 
    ],
    "fields": [
        {
            "name": "category_name",
            "selector": "h2.category-name",
            "type": "text"
        },
        {
            "name": "products",
            "selector": "div.product",
            "type": "nested_list",    # repeated sub-objects
            "fields": [
                {
                    "name": "name",
                    "selector": "h3.product-name",
                    "type": "text"
                },
                {
                    "name": "price",
                    "selector": "p.product-price",
                    "type": "text"
                },
                {
                    "name": "details",
                    "selector": "div.product-details",
                    "type": "nested",  # single sub-object
                    "fields": [
                        {
                            "name": "brand",
                            "selector": "span.brand",
                            "type": "text"
                        },
                        {
                            "name": "model",
                            "selector": "span.model",
                            "type": "text"
                        }
                    ]
                },
                {
                    "name": "features",
                    "selector": "ul.product-features li",
                    "type": "list",
                    "fields": [
                        {"name": "feature", "type": "text"} 
                    ]
                },
                {
                    "name": "reviews",
                    "selector": "div.review",
                    "type": "nested_list",
                    "fields": [
                        {
                            "name": "reviewer", 
                            "selector": "span.reviewer", 
                            "type": "text"
                        },
                        {
                            "name": "rating", 
                            "selector": "span.rating", 
                            "type": "text"
                        },
                        {
                            "name": "comment", 
                            "selector": "p.review-text", 
                            "type": "text"
                        }
                    ]
                },
                {
                    "name": "related_products",
                    "selector": "ul.related-products li",
                    "type": "list",
                    "fields": [
                        {
                            "name": "name", 
                            "selector": "span.related-name", 
                            "type": "text"
                        },
                        {
                            "name": "price", 
                            "selector": "span.related-price", 
                            "type": "text"
                        }
                    ]
                }
            ]
        }
    ]
}
```

Key Takeaways:

- **Nested vs. List**:  
  - **`type: "nested"`** means a **single** sub-object (like `details`).  
  - **`type: "list"`** means multiple items that are **simple** dictionaries or single text fields.  
  - **`type: "nested_list"`** means repeated **complex** objects (like `products` or `reviews`).
- **Base Fields**: We can extract **attributes** from the container element via `"baseFields"`. For instance, `"data_cat_id"` might be `data-cat-id="elect123"`.  
- **Transforms**: We can also define a `transform` if we want to lower/upper case, strip whitespace, or even run a custom function.

### Running the Extraction

```python
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

ecommerce_schema = {
    # ... the advanced schema from above ...
}

async def extract_ecommerce_data():
    strategy = JsonCssExtractionStrategy(ecommerce_schema, verbose=True)
    
    config = CrawlerRunConfig()
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://gist.githubusercontent.com/githubusercontent/2d7b8ba3cd8ab6cf3c8da771ddb36878/raw/1ae2f90c6861ce7dd84cc50d3df9920dee5e1fd2/sample_ecommerce.html",
            extraction_strategy=strategy,
            config=config
        )

        if not result.success:
            print("Crawl failed:", result.error_message)
            return
        
        # Parse the JSON output
        data = json.loads(result.extracted_content)
        print(json.dumps(data, indent=2) if data else "No data found.")

asyncio.run(extract_ecommerce_data())
```

If all goes well, you get a **structured** JSON array with each “category,” containing an array of `products`. Each product includes `details`, `features`, `reviews`, etc. All of that **without** an LLM.

---

## 4. Why “No LLM” Is Often Better

1. **Zero Hallucination**: Schema-based extraction doesn’t guess text. It either finds it or not.  
2. **Guaranteed Structure**: The same schema yields consistent JSON across many pages, so your downstream pipeline can rely on stable keys.  
3. **Speed**: LLM-based extraction can be 10–1000x slower for large-scale crawling.  
4. **Scalable**: Adding or updating a field is a matter of adjusting the schema, not re-tuning a model.

**When might you consider an LLM?** Possibly if the site is extremely unstructured or you want AI summarization. But always try a schema approach first for repeated or consistent data patterns.

---

## 5. Base Element Attributes & Additional Fields

It’s easy to **extract attributes** (like `href`, `src`, or `data-xxx`) from your base or nested elements using:

```json
{
  "name": "href",
  "type": "attribute",
  "attribute": "href",
  "default": null
}
```

You can define them in **`baseFields`** (extracted from the main container element) or in each field’s sub-lists. This is especially helpful if you need an item’s link or ID stored in the parent `<div>`.

---

## 6. Putting It All Together: Larger Example

Consider a blog site. We have a schema that extracts the **URL** from each post card (via `baseFields` with an `"attribute": "href"`), plus the title, date, summary, and author:

```python
schema = {
  "name": "Blog Posts",
  "baseSelector": "a.blog-post-card",
  "baseFields": [
    {"name": "post_url", "type": "attribute", "attribute": "href"}
  ],
  "fields": [
    {"name": "title", "selector": "h2.post-title", "type": "text", "default": "No Title"},
    {"name": "date", "selector": "time.post-date", "type": "text", "default": ""},
    {"name": "summary", "selector": "p.post-summary", "type": "text", "default": ""},
    {"name": "author", "selector": "span.post-author", "type": "text", "default": ""}
  ]
}
```

Then run with `JsonCssExtractionStrategy(schema)` to get an array of blog post objects, each with `"post_url"`, `"title"`, `"date"`, `"summary"`, `"author"`.

---

## 7. Tips & Best Practices

1. **Inspect the DOM** in Chrome DevTools or Firefox’s Inspector to find stable selectors.  
2. **Start Simple**: Verify you can extract a single field. Then add complexity like nested objects or lists.  
3. **Test** your schema on partial HTML or a test page before a big crawl.  
4. **Combine with JS Execution** if the site loads content dynamically. You can pass `js_code` or `wait_for` in `CrawlerRunConfig`.  
5. **Look at Logs** when `verbose=True`: if your selectors are off or your schema is malformed, it’ll often show warnings.  
6. **Use baseFields** if you need attributes from the container element (e.g., `href`, `data-id`), especially for the “parent” item.  
7. **Performance**: For large pages, make sure your selectors are as narrow as possible.

---

## 8. Schema Generation Utility

While manually crafting schemas is powerful and precise, Crawl4AI now offers a convenient utility to **automatically generate** extraction schemas using LLM. This is particularly useful when:

1. You're dealing with a new website structure and want a quick starting point
2. You need to extract complex nested data structures
3. You want to avoid the learning curve of CSS/XPath selector syntax

### Using the Schema Generator

The schema generator is available as a static method on both `JsonCssExtractionStrategy` and `JsonXPathExtractionStrategy`. You can choose between OpenAI's GPT-4 or the open-source Ollama for schema generation:

```python
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, JsonXPathExtractionStrategy
from crawl4ai.types import LLMConfig

# Sample HTML with product information
html = """
<div class="product-card">
    <h2 class="title">Gaming Laptop</h2>
    <div class="price">$999.99</div>
    <div class="specs">
        <ul>
            <li>16GB RAM</li>
            <li>1TB SSD</li>
        </ul>
    </div>
</div>
"""

# Option 1: Using OpenAI (requires API token)
css_schema = JsonCssExtractionStrategy.generate_schema(
    html,
    schema_type="css", 
    llm_config = LLMConfig(provider="openai/gpt-4o",api_token="your-openai-token")
)

# Option 2: Using Ollama (open source, no token needed)
xpath_schema = JsonXPathExtractionStrategy.generate_schema(
    html,
    schema_type="xpath",
    llm_config = LLMConfig(provider="ollama/llama3.3", api_token=None)  # Not needed for Ollama
)

# Use the generated schema for fast, repeated extractions
strategy = JsonCssExtractionStrategy(css_schema)
```

### LLM Provider Options

1. **OpenAI GPT-4 (`openai/gpt4o`)**
   - Default provider
   - Requires an API token
   - Generally provides more accurate schemas
   - Set via environment variable: `OPENAI_API_KEY`

2. **Ollama (`ollama/llama3.3`)**
   - Open source alternative
   - No API token required
   - Self-hosted option
   - Good for development and testing

### Benefits of Schema Generation

1. **One-Time Cost**: While schema generation uses LLM, it's a one-time cost. The generated schema can be reused for unlimited extractions without further LLM calls.
2. **Smart Pattern Recognition**: The LLM analyzes the HTML structure and identifies common patterns, often producing more robust selectors than manual attempts.
3. **Automatic Nesting**: Complex nested structures are automatically detected and properly represented in the schema.
4. **Learning Tool**: The generated schemas serve as excellent examples for learning how to write your own schemas.

### Best Practices

1. **Review Generated Schemas**: While the generator is smart, always review and test the generated schema before using it in production.
2. **Provide Representative HTML**: The better your sample HTML represents the overall structure, the more accurate the generated schema will be.
3. **Consider Both CSS and XPath**: Try both schema types and choose the one that works best for your specific case.
4. **Cache Generated Schemas**: Since generation uses LLM, save successful schemas for reuse.
5. **API Token Security**: Never hardcode API tokens. Use environment variables or secure configuration management.
6. **Choose Provider Wisely**: 
   - Use OpenAI for production-quality schemas
   - Use Ollama for development, testing, or when you need a self-hosted solution

That's it for **Extracting JSON (No LLM)**! You've seen how schema-based approaches (either CSS or XPath) can handle everything from simple lists to deeply nested product catalogs—instantly, with minimal overhead. Enjoy building robust scrapers that produce consistent, structured JSON for your data pipelines!

---

## 9. Conclusion

With **JsonCssExtractionStrategy** (or **JsonXPathExtractionStrategy**), you can build powerful, **LLM-free** pipelines that:

- Scrape any consistent site for structured data.  
- Support nested objects, repeating lists, or advanced transformations.  
- Scale to thousands of pages quickly and reliably.

**Next Steps**:

- Combine your extracted JSON with advanced filtering or summarization in a second pass if needed.  
- For dynamic pages, combine strategies with `js_code` or infinite scroll hooking to ensure all content is loaded.

**Remember**: For repeated, structured data, you don’t need to pay for or wait on an LLM. A well-crafted schema plus CSS or XPath gets you the data faster, cleaner, and cheaper—**the real power** of Crawl4AI.

**Last Updated**: 2025-01-01

---

That’s it for **Extracting JSON (No LLM)**! You’ve seen how schema-based approaches (either CSS or XPath) can handle everything from simple lists to deeply nested product catalogs—instantly, with minimal overhead. Enjoy building robust scrapers that produce consistent, structured JSON for your data pipelines!
```
