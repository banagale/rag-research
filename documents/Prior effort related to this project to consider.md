I worked on two proofs of concept leading up to this:

1.  A direct extraction of pages from Microsoft product firmware pages.
2. An iterative python template generator for security bulletin pages from AMD.

Both are housed in the `crawlers-ai` repository at: https://gitlab.com/fproj/data-team/crawlers-ai

There are other supporting files, but these should help you infer ideas (and possibly code examples) to bring into the new project.

# Sample code from microsoft product firmware

## /My Drive/code/crawlers-ai/crawlers_ai/vendors/microsoft.py

```python
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

from crawlers_ai.metadata_fields import FileTypes


class FirmwareMetadata(BaseModel):
    binary_name: str
    file_url: str
    date: str
    version: str
    status: str
    file_type: str
    model: str
    product_url: str


def get_url(file_url: str) -> str | None:
    """
    Returns a valid URL from unexpectedly broken output of crawl4ai extracted_fields
    Example: "https://www.microsoft.com/download/<https:/download.microsoft.com/download/4/C/7/4C7DA85D-03DE-4B89-B8E0-386437331C46/SurfacePro_Win11_22000_24.010.4410.0.msi>"
    Created to handle unexpected output from crawl4ai
    https://github.com/unclecode/crawl4ai/issues/583 fixed the urls as parsed but broke extraction in a different way
    """
    match = re.search(r"<(https?:/[^<>\s]+)>", file_url)
    if match:
        return match.group(1).replace("https:/", "https://", 1)
    return None


class MicrosoftExtractor:
    def __init__(self, url: str):
        self.url = url

    async def extract(self):
        """
        Extract firmware metadata using LLM-powered Crawl4AI.
        """
        logging.info("Starting LLM-based extraction for Microsoft firmware page: %s", self.url)

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logging.error("OPENAI_API_KEY not set")
            return

        instruction = """Extract firmware metadata as structured JSON.
        Ensure each field is extracted accurately from the page content.
        Return only valid data, without hallucinating missing information.

        Fields to extract:
        - `binary_name` (string): Name of the firmware file.
        - `file_url` (string): **Direct firmware download link.** This appears inside a download modal and should should have the newest version number. 
        - `date` (string, format: YYYY-MM-DD hh:mm:ss A): Date and time the firmware was published. Ensure the extracted datetime strictly follows the `YYYY-MM-DD hh:mm:ss A` format, using 12-hour notation with `AM` or `PM`.
        - `version` (string): Firmware version, extracted from metadata or filename.
        - `file_type` (string): Type of firmware (e.g., BIOS, firmware, driver).
        - `model` (string): Device model the firmware applies to.
        - `product_url` (string): URL of the product support/download page.

        Output example:
        ```json
        [
            {
                "binary_name": "SurfacePro_Win10_19044_24.010.4410.0.msi",
                "file_url": "https://download.microsoft.com/example.msi",
                "date": "2024-01-13 1:40:30 AM",
                "version": "24.010.4410.0",
                "file_type": "BIOS",
                "model": "Surface Pro 5",
                "product_url": "https://www.microsoft.com/download/details.aspx?id=55484"
            }
        ]
        """  # noqa: E501

        # Define LLM extraction strategy
        llm_strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o",
            api_token=openai_key,
            schema=FirmwareMetadata.model_json_schema(),
            extraction_type="schema",
            instruction=instruction,
            chunk_token_threshold=1200,  # Split long pages for efficiency
            overlap_rate=0.1,
            apply_chunking=True,
            input_format="markdown",  # LLM processes markdown input
            extra_args={"temperature": 0.1, "max_tokens": 1000},
            verbose=True,
        )

        excluded_selectors = (
            "header, footer, nav, .social-media, .share-links, .social-links, .socialfollow, "
            'section[aria-label*="banner"], .back-to-top-button, .m-skip-to-main'
        )
        # Configure crawler
        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            cache_mode=CacheMode.DISABLED,
            exclude_social_media_links=True,
            excluded_tags=["script", "style", "form", "header", "footer", "nav"],
            excluded_selector=excluded_selectors,
            magic=True,
            js_code="document.querySelector('button.dlcdetail__download-btn')?.click();",
            # wait_for="css:.dlc-details-multi-file-download-window",
            delay_before_return_html=1.5,
        )

        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            result = await crawler.arun(self.url, config=crawl_config)
            if not result.success:
                logging.error("Extraction failed for %s: %s", self.url, result.error_message)
                return None

            # Parse extracted data
            try:
                extracted_data = json.loads(result.extracted_content)
                if not isinstance(extracted_data, list):
                    logging.error("Unexpected format: extracted content is not a list.")
                    return None
            except json.JSONDecodeError:
                logging.error("Failed to parse extracted content as JSON.")
                return None

            grouped_by_model = defaultdict(list)

            for record in extracted_data:
                # Hack to fix file_urls
                record["file_url"] = get_url(record.get("file_url", ""))
                # Manual override of type, as all microsoft integrity are BIOS
                record["file_type"] = FileTypes.BIOS
                # Stub missing fields
                record["status"] = ""
                record["local_name"] = ""
                record["location_path"] = ""
                record["md5"] = ""
                record["sha1"] = ""
                record["sha256"] = ""
                record["downloaded"] = ""
                record["file_url_is_valid"] = ""

                model = record.get("model", "Unknown")
                grouped_by_model[model].append(record)

            logging.info("LLM Extraction completed. %d records found.", len(extracted_data))

            final_results = []
            for model, records in grouped_by_model.items():
                # Sort records by most recent published date
                records.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %I:%M:%S %p"), reverse=True)

                # Keep only the most recent firmware version per model
                latest_record = records[0]
                final_results.append(latest_record)

            return final_results

```
## /My Drive/code/crawlers-ai/crawlers_ai/core.py

```python
import asyncio
import json
import logging

from crawlers_ai.vendors.microsoft import MicrosoftExtractor


async def extract_firmware_metadata(url: str):
    """
    Extract firmware metadata from a vendor's support page using the LLM-powered Crawl4AI pipeline.
    """
    logging.info("Extracting firmware metadata from URL: %s", url)

    extractor = MicrosoftExtractor(url)
    data = await extractor.extract()

    if data is None:
        logging.warning("No data extracted from %s", url)
        return []

    logging.info("Extraction completed successfully. Records extracted: %d", len(data))
    return data


if __name__ == "__main__":
    firmware_url = "https://www.microsoft.com/download/details.aspx?id=55484"
    firmware_metadata = asyncio.run(extract_firmware_metadata(firmware_url))
    print(json.dumps(firmware_metadata, indent=4))

```



# Sample code from iterative python template generator for security bulletin pages from AMD

## /My Drive/code/crawlers-ai/scripts/amd-spike/macrodata-refinement.py

```python
import asyncio
import itertools
import json
import logging
import re
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from litellm import RateLimitError, completion

BASE_DIR = Path("crawlers_ai/tmp")
LOG_FILE = BASE_DIR / "compare_crawlers.log"
GENERATED_CODE_DIR = BASE_DIR / "amd_generated_code"

HTML_FILE = BASE_DIR / "amd_firmware_pages/amd-sb-3019.html"
REFERENCE_JSON = BASE_DIR / "amd_benchmark_data/amd-nongenerated.json"
REFERENCE_JSON_KEY_EXCLUSIONS = {
    "provenance",
    "confscore",
    "component",
    "references",
    "severity",
    "attackVector",
    "score",
}  # noqa E501


def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)


def create_run_directory():
    old_dir = GENERATED_CODE_DIR / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    # Move any existing run_... directories to 'old' before creating a new run_... directory.
    for item in GENERATED_CODE_DIR.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            shutil.move(str(item), str(old_dir / item.name))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = GENERATED_CODE_DIR / f"run_{run_id}"
    html_sources_dir = run_dir / "html_sources"
    prompts_dir = run_dir / "prompts"
    run_dir.mkdir(parents=True, exist_ok=True)
    html_sources_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Beginning main iterative code generation loop. Storing output in {run_dir}")
    return run_dir, html_sources_dir, prompts_dir


def estimate_tokens(text):
    return len(text.split()) if text else 0


async def preprocess_html(html_path: Path):
    with open(html_path, "r", encoding="utf-8") as file:
        raw_html = file.read()
    logging.info("Estimating token count before processing...")
    tokens_before = estimate_tokens(raw_html)
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        exclude_social_media_links=True,
        excluded_tags=["script", "style", "form", "header", "footer", "nav"],
        magic=True,
    )
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        result = await crawler.arun(f"raw:{raw_html}", config=crawl_config)
    if not result.success:
        logging.error("HTML preprocessing failed: %s", result.error_message)
        return raw_html, None
    preprocessed_html = result.cleaned_html or ""
    tokens_after = estimate_tokens(preprocessed_html)
    logging.info(
        f"Token count before: {tokens_before}, after: {tokens_after}, "
        f"reduction: {((tokens_before - tokens_after) / max(tokens_before, 1)) * 100:.2f}%"
    )
    return raw_html, preprocessed_html


def validate_preprocessed_output(extracted_content):
    try:
        with open(REFERENCE_JSON, "r", encoding="utf-8") as file:
            reference_data = json.load(file)
        missing_values = []
        for key, value in reference_data.items():
            if key in REFERENCE_JSON_KEY_EXCLUSIONS:
                continue
            if isinstance(value, (int, float)):
                value_str = str(value)
                if value_str not in extracted_content:
                    missing_values.append(f"{key}: {value_str}")
            elif isinstance(value, str):
                if value and value not in extracted_content:
                    missing_values.append(f"{key}: {value}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item and item not in extracted_content:
                        missing_values.append(f"{key} (list item): {item}")
        if missing_values:
            logging.warning("Validation failed: Missing key values: {missing_values}")
            return False
        logging.info("Validation successful: All expected key values found in preprocessed output.")
        return True
    except Exception as e:
        logging.error(f"Error validating preprocessed output: {str(e)}")
        return False


def save_html_files(html_sources_dir, raw_html, preprocessed_html):
    raw_html_path = html_sources_dir / "amd-sb-3019.html"
    preprocessed_html_path = html_sources_dir / "amd-sb-3019_preprocessed.html"
    raw_html_path.write_text(raw_html, encoding="utf-8")
    preprocessed_html_path.write_text(preprocessed_html, encoding="utf-8")
    return raw_html_path, preprocessed_html_path


def save_metadata_json(run_dir, html_sources_dir, raw_html_path, preprocessed_html_path):
    json_output = {
        "base_run_dir": str(run_dir.name),
        "raw_html_file": f"./{html_sources_dir.name}/{raw_html_path.name}",
        "preprocessed_html_file": f"./{html_sources_dir.name}/{preprocessed_html_path.name}",
    }
    json_path = run_dir / "processed_amd_sb_3019.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_output, file, indent=4)
    logging.info(f"Saved JSON metadata: {json_path}")


def clean_generated_code(raw_code):
    match = re.search(r"```python\n(.*?)\n```", raw_code, re.DOTALL)
    return match.group(1) if match else raw_code


def validate_script(script_path: Path):
    try:
        with script_path.open("r", encoding="utf-8") as file:
            compile(file.read(), str(script_path), "exec")
        logging.info(f"Validation successful: {script_path} compiles correctly.")
        return True
    except SyntaxError as e:
        logging.error(f"Validation failed: {script_path} contains syntax errors: {e}")
        return False


def save_prompt(prompts_dir, iteration, prompt):
    filename = f"prompt_{iteration:03d}.txt"
    prompt_path = prompts_dir / filename
    with open(prompt_path, "w", encoding="utf-8") as file:
        file.write(prompt)
    logging.info(f"Saved prompt: {prompt_path}")


def generate_prompt(keys_to_extract, reference_json, preprocessed_html, successful_selectors=None, feedback=None):
    if successful_selectors is None:
        successful_selectors = {}
    prompt = f"""
        You are an AI tasked with generating a Python script to extract structured metadata from security bulletins.

        #### **Instructions**
        - Study the keys in the provided **Reference JSON**.
        - Examine **where** the values of these keys appear in the **Preprocessed HTML** below.
        - Identify the surrounding HTML structure and derive **XPath selectors** or **CSS selectors** to locate each key’s value dynamically.
        - Ensure each selector successfully extracts non-empty values when applied to the HTML.
        - Format the extracted data into structured JSON, matching the reference format for the keys: {", ".join(keys_to_extract)}.
        - Save the output to `'./new_out.json'`.

        #### **Output Requirements**
        - Generate **fully functional** Python code.
        - Use **lxml** for parsing.
        - Read the HTML from `./html_sources/amd-sb-3019_preprocessed.html`.
        - Validate and log all extracted values (use Python's logging module).
        - If a value is missing, attempt alternative parsing strategies.
        - The script must be modular and reusable.

        #### **Constraints**
        - Do **not** hardcode specific values.
        - Do **not** assume the HTML structure is static.
        - The script should be robust to minor HTML changes.

        #### **Testing & Logging**
        - Ensure all selectors **return valid data** before writing JSON.
        - Log **success/failure rates** per key to measure script accuracy.

        #### **Reference JSON**
        ```json
            {json.dumps(reference_json, indent=4)}
        ```

        #### **Preprocessed HTML**
        ```html
        {preprocessed_html}
        ```
    """  # noqa E501
    if successful_selectors:
        prompt += f"""
            #### **Successful Selectors (reuse these)**
            Reuse these exact selectors for the listed fields without modification:
            ```python
            {json.dumps(successful_selectors, indent=4)}
            ```
            - Generate new selectors only for fields not listed above.
        """
    if feedback:
        prompt += f"""
            #### **Feedback**
            {feedback}
        """
    else:
        prompt += """
            #### **Feedback**
            No feedback provided.
        """
    return prompt


def generate_llm_extraction_script(
    run_dir, prompts_dir, iteration, preprocessed_html, successful_selectors=None, feedback=None
):
    with open(REFERENCE_JSON, "r", encoding="utf-8") as file:
        reference_json = json.load(file)
    keys_to_extract = [k for k in reference_json if k not in REFERENCE_JSON_KEY_EXCLUSIONS]
    prompt = generate_prompt(keys_to_extract, reference_json, preprocessed_html, successful_selectors, feedback)
    save_prompt(prompts_dir, iteration, prompt)

    # Initialize variables for streaming
    raw_code = ""
    spinner = itertools.cycle(["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"])

    print("Starting LLM request...", flush=True)
    print("\033[?25l", end="", flush=True)  # Hide cursor

    try:
        # Use stream=True parameter to enable streaming
        stream_response = completion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=5000, stream=True
        )

        # Begin streaming loop
        chunk_count = 0

        start_time = time.time()
        # This is the streaming loop - we iterate through each chunk as it arrives
        for chunk in stream_response:
            # Extract content from chunk
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"].get("content")
                    if content is not None:  # Add this check to avoid NoneType error
                        raw_code += content
                        chunk_count += 1
                        if chunk_count % 15 == 0:
                            elapsed = time.time() - start_time
                            # print("\rGenerating code ", end="", flush=True)
                            print(
                                f"\rReceiving LLM response... "
                                f"{elapsed:.1f}s elapsed, {len(raw_code)} chars {next(spinner)}",
                                end="",
                                flush=True,
                            )
        # End streaming loop

        print("\nLLM response complete!", flush=True)

    except RateLimitError as e:
        logging.error(f"Rate limit exceeded: {e}")
        return None

    cleaned_code = clean_generated_code(raw_code)
    script_path = run_dir / f"gpt4.{iteration:03d}.py"
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(cleaned_code)
    logging.info(f"Generated extraction script: {script_path}")
    if not validate_script(script_path):
        logging.error(f"Generated script {script_path} failed validation.")
        return None
    return script_path


# Changes to execute_script function
def execute_script(script_path):
    run_dir = script_path.parent
    script_name = script_path.name
    output_json = run_dir / "new_out.json"
    if output_json.exists():
        output_json.unlink()  # Remove previous output to ensure fresh run
    try:
        result = subprocess.run(
            ["python", script_name],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        logging.info(f"Script {script_path} executed with this result: {result.stdout}")
        return output_json, result.stdout, None  # Added None for error_message
    except subprocess.CalledProcessError as e:
        error_message = e.stderr
        # Extract the specific error type and message
        error_type = "Unknown error"
        if "Traceback" in error_message:
            error_lines = error_message.strip().split("\n")
            for line in reversed(error_lines):
                if ": " in line and not line.startswith(" "):
                    error_type = line.strip()
                    break
        logging.error(f"Script {script_path} failed with error: {error_type}")
        return None, None, error_type  # Return error message


def compare_extracted_data(extracted_json_path, reference_json_path):
    try:
        with open(extracted_json_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)
        with open(reference_json_path, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON files: {e}")
        return 0, [], []
    keys_to_compare = [k for k in reference_data if k not in REFERENCE_JSON_KEY_EXCLUSIONS]
    correct_fields = []
    incorrect_fields = []
    for key in keys_to_compare:
        ref_value = reference_data.get(key)
        ext_value = extracted_data.get(key)
        ref_str = str(ref_value) if ref_value is not None else ""
        ext_str = str(ext_value) if ext_value is not None else ""
        if ref_str == ext_str:
            correct_fields.append(key)
        else:
            incorrect_fields.append(key)
    total_fields = len(keys_to_compare)
    accuracy = len(correct_fields) / total_fields if total_fields > 0 else 0
    return accuracy, correct_fields, incorrect_fields


def log_results(iteration, accuracy, correct_fields, incorrect_fields):
    logging.info(f"Iteration {iteration}: Accuracy {accuracy:.2%}")
    logging.info(f"Correct fields: {', '.join(correct_fields)}")
    logging.info(f"Incorrect fields: {', '.join(incorrect_fields)}")


def prepare_feedback(correct_fields, incorrect_fields, preprocessed_html, stdout):
    if not incorrect_fields:
        return "All fields extracted correctly."
    feedback = f"The following fields were extracted incorrectly: {', '.join(incorrect_fields)}. "
    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        reference_data = json.load(f)
    expected_values = {field: reference_data[field] for field in incorrect_fields if field in reference_data}
    feedback += (
        "Please refine the selectors or extraction logic for these fields "
        "based on the preprocessed HTML to match the expected values:\n"
    )
    for field, value in expected_values.items():
        feedback += f"- {field}: Expected '{value}'\n"
    feedback += "Review the **Preprocessed HTML** to locate these values and adjust the selectors accordingly."
    return feedback


def extract_selector_from_stdout(stdout, field):
    # Extract selector from script output assuming the format: "Extracted {field} using {selector}: {value}"
    pattern = rf"Extracted {field} using (.+?):"
    match = re.search(pattern, stdout)
    if match:
        return match.group(1)
    logging.warning(f"No selector found for {field} in stdout.")
    return None


async def main():
    setup_logging()
    run_dir, html_sources_dir, prompts_dir = create_run_directory()
    raw_html, preprocessed_html = await preprocess_html(HTML_FILE)
    if not validate_preprocessed_output(preprocessed_html):
        logging.error("Preprocessing validation failed. Exiting.")
        return
    raw_html_path, preprocessed_html_path = save_html_files(html_sources_dir, raw_html, preprocessed_html)
    save_metadata_json(run_dir, html_sources_dir, raw_html_path, preprocessed_html_path)

    iteration = 1
    previous_accuracy = 0
    improvement = 0
    feedback = None
    successful_selectors = {}

    # Track last successful script and best script
    last_successful_script = None  # Any script that runs without errors
    last_successful_json = None
    last_successful_stdout = None

    # Track best performing script
    best_script = None  # Script with highest accuracy
    best_json = None
    best_stdout = None
    best_accuracy = 0

    while True:
        script_path = generate_llm_extraction_script(
            run_dir, prompts_dir, iteration, preprocessed_html, successful_selectors, feedback
        )
        if script_path is None:
            logging.error("Failed to generate script. Exiting.")
            break

        extracted_json_path, stdout, error_message = execute_script(script_path)

        if extracted_json_path is None or not extracted_json_path.exists():
            logging.error(f"Script execution failed in iteration {iteration}. Error: {error_message}")

            if improvement < 0 and best_script and best_script != last_successful_script:
                # For negative improvement, use best script instead of last successful
                logging.info(
                    f"Negative improvement or error detected. Falling back to BEST script with "
                    f"accuracy {best_accuracy:.2%}"
                )
                extracted_json_path = best_json
                stdout = best_stdout
                script_path = best_script  # Use the best script for future reference
            elif last_successful_script:
                logging.info(f"Falling back to last successful script from iteration {iteration - 1}")
                # Use last successful data
                extracted_json_path = last_successful_json
                stdout = last_successful_stdout

                # Add error information to feedback
                if feedback:
                    feedback += (
                        f"\n\nYour previous script failed with error: {error_message}. Please fix this issue "
                        f"while maintaining the successful extractions from previous iterations."
                    )
                else:
                    feedback = (
                        f"Your script failed with error: {error_message}. Please fix this issue while "
                        f"maintaining the successful extractions from previous iterations."
                    )
            else:
                logging.error("No successful script found from previous iterations. Setting accuracy to 0.")
                accuracy, correct_fields, incorrect_fields = 0, [], []

                # Create specific feedback about the error
                feedback = (
                    f"Your script failed with error: {error_message}. Please make your code more robust with"
                    f" proper error handling and validation before attempting operations."
                )

                # Continue to next iteration
                iteration += 1
                continue

        # If we got here with extracted_json_path, we either had a successful run or used a fallback
        versioned_json_path = run_dir / f"new_out_{iteration:03d}.json"

        # Copy if the file exists (could be using a fallback file)
        if extracted_json_path and Path(extracted_json_path).exists():
            shutil.copy(extracted_json_path, versioned_json_path)
            accuracy, correct_fields, incorrect_fields = compare_extracted_data(extracted_json_path, REFERENCE_JSON)

            # If this was a successful run (not a fallback), update our last successful references
            if error_message is None:
                last_successful_script = script_path
                last_successful_json = extracted_json_path
                last_successful_stdout = stdout

                # Track best performing script
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_script = script_path
                    best_json = extracted_json_path
                    best_stdout = stdout
                    logging.info(f"New best script from iteration {iteration} with accuracy {accuracy:.2%}")
                    # Save a copy marked as best
                    best_versioned_path = run_dir / "new_out_best.json"
                    shutil.copy(extracted_json_path, best_versioned_path)
        else:
            # Should not reach here with the new logic, but just in case
            accuracy, correct_fields, incorrect_fields = 0, [], []

        log_results(iteration, accuracy, correct_fields, incorrect_fields)
        if accuracy == 1:
            logging.info("Perfect accuracy achieved! Stopping.")
            break
        if accuracy > 0.9:
            logging.info("Accuracy in excess of 90% achieved. Stopping.")
            break

        improvement = accuracy - previous_accuracy
        progress = "N/A" if iteration == 1 else f"{improvement:.2%}"

        # Display prompt
        if improvement >= 0:
            continuation_status = "(will automatically continue in five seconds)"
        else:
            continuation_status = "(will not continue automatically)"
        print(
            f"Accuracy: {accuracy:.2%} (Improvement: {progress}). \nyContinue refining? (y/n) {continuation_status}: ",
            end="",
            flush=True,
        )

        def get_input_with_timeout(timeout=5):
            """Get user input with a timeout."""
            # Only works on Unix-like systems
            if hasattr(select, "select"):
                i, o, e = select.select([sys.stdin], [], [], timeout)
                if i:
                    return sys.stdin.readline().strip()
            # Fallback for Windows
            else:
                import msvcrt

                start_time = time.time()
                input_str = ""
                while time.time() - start_time < timeout:
                    if msvcrt.kbhit():
                        char = msvcrt.getche().decode("utf-8")
                        if char == "\r":  # Enter key
                            print("")
                            break
                        input_str += char
                    time.sleep(0.1)
                return input_str
            return None

        # Handle input based on improvement
        if improvement < 0:
            # For negative improvement: wait indefinitely for input
            user_input = input()
        else:
            # For positive improvement: use timeout
            user_input = get_input_with_timeout(5)
            if user_input is None:
                user_input = "y"

        # Input validation
        if not user_input or user_input.lower() not in ["y", "n"]:
            print("Invalid input. Defaulting to continue.")
            user_input = "y"

        if user_input.lower() == "n":
            logging.info("User chose to stop.")
            break

        # Update successful selectors based on correct fields
        for field in correct_fields:
            if field not in successful_selectors:
                selector = extract_selector_from_stdout(stdout, field)
                if selector:
                    successful_selectors[field] = selector

        # Prepare feedback with error information already included if there was an error
        if error_message is None:
            feedback = prepare_feedback(correct_fields, incorrect_fields, preprocessed_html, stdout)

        previous_accuracy = accuracy
        iteration += 1


if __name__ == "__main__":
    asyncio.run(main())

```
## /My Drive/code/crawlers-ai/scripts/amd-spike/readme.md

```plaintext
1. Set up the project dependencies according to the main readme
2. Don't forget the step from 1: set the `OPENAI_API_KEY` environment variable before running the script.
3. From 
[this folder on GCS](https://console.cloud.google.com/storage/browser/rob_banagale_work/amd-spike?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)), copy `sample-data`'s contents to `crawlers_ai/tmp/` before running .
4. Run the script generation script from the project root:

```bash
    poetry run python ./scripts/amd-spike/macrodata-refinement.py crawlers_ai/tmp/amd_firmware_pages ./crawlers_ai/tmp/new_out.json
```
```
