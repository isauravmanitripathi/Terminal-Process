#!/usr/bin/env python3
"""Jupyter Notebook Code Explainer - Async Batch Processing Edition

Features:
- Parallel Processing (AsyncIO) for high speed.
- Automatic removal of empty cells.
- Single file, Folder, and Nested Folder support.
- Detailed terminal logging.
"""

import os
import json
import time
import argparse
import gc
import asyncio
from pathlib import Path
from openai import AsyncOpenAI, APIError, RateLimitError
from dotenv import load_dotenv
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-5-mini"  # Using a cost-effective model
MAX_RETRIES = 3
RETRY_DELAY = 2
CONCURRENT_REQUESTS = 10  # How many API calls to run in parallel (Speed vs Rate Limit)

# === PROMPTS ===
CODE_EXPLANATION_PROMPT = """You are a knowledgeable Team Lead explaining this code to a developer. Write a clear, paragraph-based explanation that balances technical depth with readability.

**Guidelines:**
1. **Narrative Walkthrough:** Explain the logic sequentially. Tell the "story" of how the data flows through this block and what decisions the code makes.
2. **Focus on "Why" and "How":** Do not just say *what* is happening; explain *why* it is happening (e.g., "We normalize the data here to prevent the gradients from exploding during training").
3. **Skip Trivial Syntax:** Assume the reader knows Python. Do not explain basic syntax like imports or variable assignments. Focus on the algorithms and business logic.
4. **Professional Tone:** Use clear, professional language. Avoid analogies or overly casual slang, but stay warm and helpful.
5. **Context:** Tie the specific actions back to the overall goal of: {project_name}.

Code to explain:
```python
{code_content}
```
Output only the explanation."""

MARKDOWN_REWRITE_PROMPT = """Refine and polish the markdown text below to make it read like high-quality technical documentation.
- Improve the flow, clarity, and sentence structure.
- Remove awkward phrasing or unnecessary jargon.
- Make it sound professional yet accessible.
- **Do not** add new information or change the underlying meaning.

Original text:
{markdown_content}

Return only the rewritten text."""

EXPLANATION_PROMPT = CODE_EXPLANATION_PROMPT
MARKDOWN_PROMPT = MARKDOWN_REWRITE_PROMPT


class NotebookExplainer:
    """Process Jupyter notebooks: explain code cells and simplify markdown cells (Async)"""

    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        # Initialize Async OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = AsyncOpenAI(api_key=api_key)
        print(f"✓ Initialized Async Notebook Explainer (Model: {self.model})")

    def read_notebook(self, notebook_path: str) -> dict:
        """Read and parse Jupyter notebook JSON"""
        path = Path(notebook_path)
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        if path.suffix != '.ipynb':
            raise ValueError(f"File must be a Jupyter notebook (.ipynb): {notebook_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            return notebook_data
        except Exception as e:
            raise Exception(f"Error reading notebook: {e}")

    def clean_empty_cells(self, notebook_data: dict) -> Tuple[dict, int]:
        """Removes empty code or markdown cells"""
        original_cells = notebook_data.get('cells', [])
        cleaned_cells = []
        for cell in original_cells:
            source_content = "".join(cell.get('source', [])).strip()
            # Only keep if content exists
            if source_content:
                cleaned_cells.append(cell)
        
        removed_count = len(original_cells) - len(cleaned_cells)
        notebook_data['cells'] = cleaned_cells
        return notebook_data, removed_count

    def get_processable_cells(self, notebook_data: dict) -> list:
        """Extract all code and markdown cells from notebook"""
        cells = notebook_data.get('cells', [])
        processable_cells = []
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type')
            if cell_type in ['code', 'markdown']:
                content = ''.join(cell.get('source', []))
                if content.strip():
                    processable_cells.append({
                        'index': i,
                        'type': cell_type,
                        'content': content,
                        'execution_count': cell.get('execution_count', None) if cell_type == 'code' else None
                    })
        return processable_cells

    async def explain_code_async(self, code_content: str, project_name: str) -> Optional[str]:
        """Get explanation from OpenAI API for code cells (Async)"""
        prompt = EXPLANATION_PROMPT.format(project_name=project_name, code_content=code_content)
        return await self._make_api_call(prompt, "You are an expert technical writer.")

    async def simplify_markdown_async(self, markdown_content: str, project_name: str) -> Optional[str]:
        """Get simplified version from OpenAI API for markdown cells (Async)"""
        prompt = MARKDOWN_PROMPT.format(markdown_content=markdown_content)
        return await self._make_api_call(prompt, "You are an expert technical editor.")

    async def _make_api_call(self, prompt: str, system_role: str) -> Optional[str]:
        """Helper to handle async API calls with retries"""
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": prompt}
                    ]
                )
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
            except RateLimitError:
                wait_time = RETRY_DELAY * (attempt + 1)
                # print(f"\n   ⚠ Rate limit hit. Cooling down for {wait_time}s...", end='')
                await asyncio.sleep(wait_time)
            except APIError as e:
                print(f"\n   ⚠ API Error: {e}. Retrying...")
                await asyncio.sleep(RETRY_DELAY)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"\n   ✗ Failed after retries: {e}")
                    return None
        return None

    async def process_cell_task(self, semaphore: asyncio.Semaphore, cell: dict, project_name: str, progress_callback) -> dict:
        """Worker function to process a single cell under semaphore control"""
        async with semaphore:
            result = {
                'index': cell['index'],
                'type': cell['type'],
                'original_content': cell['content'],
                'success': False,
                'output': None,
                'error': None
            }
            
            try:
                if cell['type'] == 'code':
                    explanation = await self.explain_code_async(cell['content'], project_name)
                    if explanation:
                        result['success'] = True
                        result['output'] = explanation
                    else:
                        result['error'] = "Empty response from API"

                elif cell['type'] == 'markdown':
                    simplified = await self.simplify_markdown_async(cell['content'], project_name)
                    if simplified:
                        result['success'] = True
                        result['output'] = simplified
                    else:
                        result['error'] = "Empty response from API"
            
            except Exception as e:
                result['error'] = str(e)
            
            # Update progress
            progress_callback(result['success'])
            return result

    def save_notebook(self, notebook_data: dict, output_path: str) -> bool:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"   ✗ Failed to save: {e}")
            return False

    async def process_notebook_async(self, notebook_path: str, project_name: str, output_path: str = None, 
                                   show_detailed_progress: bool = True, notebook_number: Optional[int] = None, 
                                   total_notebooks: Optional[int] = None) -> Tuple[bool, List[Dict]]:
        """Main Async processing function"""
        
        input_path = Path(notebook_path)
        if input_path.stem.lower().endswith('_explained'):
            if show_detailed_progress:
                print(f"\n⚠ Skipping output file: {input_path.name}")
            return True, []

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_explained{input_path.suffix}"
            output_path = str(output_path)
        
        # 1. Read and Clean
        try:
            notebook_data = self.read_notebook(notebook_path)
            notebook_data, removed_count = self.clean_empty_cells(notebook_data)
        except Exception as e:
            print(f"✗ Failed to read notebook: {e}")
            return False, []
        
        processable_cells = self.get_processable_cells(notebook_data)
        if not processable_cells:
            print(f"\n⚠ Skipping {input_path.name} (No content cells found)")
            return True, []
        
        # 2. Display Info
        if show_detailed_progress:
            prefix = f"[{notebook_number}/{total_notebooks}] " if notebook_number else ""
            print(f"\n{prefix}📓 {input_path.name}")
            print(f"   ➤ Found {len(processable_cells)} cells to process.")
            if removed_count > 0:
                print(f"   ➤ Cleaned {removed_count} empty cells.")

        # 3. Prepare Tasks
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        stats = {'processed': 0, 'success': 0, 'failed': 0, 'total': len(processable_cells)}
        
        def update_progress(success):
            stats['processed'] += 1
            if success: stats['success'] += 1
            else: stats['failed'] += 1
            if show_detailed_progress:
                filled = int(30 * stats['processed'] / stats['total'])
                bar = '█' * filled + '░' * (30 - filled)
                print(f"\r   ➤ Processing: [{bar}] {stats['processed']}/{stats['total']} | ✓ {stats['success']} ✗ {stats['failed']}", end='', flush=True)

        # 4. Execute Async
        tasks = [
            self.process_cell_task(semaphore, cell, project_name, update_progress)
            for cell in processable_cells
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        if show_detailed_progress:
            print() # Newline after progress bar

        # 5. Reconstruct Notebook
        # Note: We modify the JSON structure here.
        # Order matters: Markdown replacements first, then Insertions (Reverse Order)
        
        failures = []
        
        # A. Apply Markdown Updates (In-place replacement)
        for res in results:
            if not res['success']:
                failures.append({'cell_index': res['index'], 'cell_type': res['type'], 'error': res['error']})
                continue
            
            if res['type'] == 'markdown':
                cell = notebook_data['cells'][res['index']]
                cell['source'] = [res['output']]
                if 'metadata' not in cell: cell['metadata'] = {}
                cell['metadata']['ai_simplified'] = True

        # B. Apply Code Explanations (Insertions)
        # Insertions shift indices, so we MUST insert from bottom to top (Reverse Index Order)
        code_results = sorted(
            [r for r in results if r['success'] and r['type'] == 'code'],
            key=lambda x: x['index'],
            reverse=True
        )
        
        for res in code_results:
            explanation_cell = {
                "cell_type": "markdown",
                "metadata": {"ai_generated": True, "explanation": True},
                "source": [res['output']]
            }
            # Insert AFTER the code cell
            notebook_data['cells'].insert(res['index'] + 1, explanation_cell)

        # 6. Save
        if self.save_notebook(notebook_data, output_path):
            print(f"   ➤ Saved to: {Path(output_path).name}")
        
        # Cleanup
        self._cleanup_notebook_memory(notebook_data, processable_cells, results)
        
        return True, failures

    def _cleanup_notebook_memory(self, *objects):
        for obj in objects:
            del obj
        gc.collect()


class BatchProcessor:
    """Handle batch processing of multiple notebooks"""
    
    def __init__(self, explainer: NotebookExplainer):
        self.explainer = explainer
    
    def scan_notebooks(self, folder_path: str, nested: bool = False) -> List[Path]:
        folder = Path(folder_path)
        if not folder.exists(): raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        glob_pattern = '*.ipynb' if not nested else '**/*.ipynb'
        notebooks = list(folder.glob(glob_pattern)) if not nested else list(folder.rglob(glob_pattern))
        
        # Filter out checkpoints and explained files
        filtered = [
            nb for nb in notebooks 
            if '.ipynb_checkpoints' not in str(nb) 
            and not nb.name.lower().endswith('_explained.ipynb')
        ]
        return sorted(filtered)
    
    def check_already_processed(self, notebook_path: Path) -> bool:
        explained = notebook_path.parent / f"{notebook_path.stem}_explained{notebook_path.suffix}"
        return explained.exists()

    async def process_batch_async(self, notebooks: List[Path], project_name: str, force: bool = False):
        to_process = [nb for nb in notebooks if force or not self.check_already_processed(nb)]
        
        if not to_process:
            print("\n✓ All notebooks in this location are already processed.")
            print("  (Use --force to re-process them if needed)")
            return

        print(f"\n╔══════════════════════════════════════════════════════╗")
        print(f"║   Starting Async Batch Processing ({len(to_process)} files)       ║")
        print(f"╚══════════════════════════════════════════════════════╝")
        
        failure_logs = []
        for i, nb in enumerate(to_process, 1):
            success, failures = await self.explainer.process_notebook_async(
                str(nb), project_name, notebook_number=i, total_notebooks=len(to_process)
            )
            
            if failures:
                log_path = self.log_failure(nb, failures)
                failure_logs.append((nb, log_path))
        
        print(f"\n{'━' * 60}")
        print("Batch Complete.")
        if failure_logs:
            print(f"⚠ Failures logged for {len(failure_logs)} notebooks. Check *_failures.json files.")

    def log_failure(self, notebook_path: Path, failures: List[Dict]):
        log_path = notebook_path.parent / f"{notebook_path.stem}_failures.json"
        data = {'notebook': str(notebook_path), 'timestamp': datetime.now().isoformat(), 'failures': failures}
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Async Jupyter Notebook Explainer")
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-n", "--notebook", type=str, help="Single notebook file")
    mode.add_argument("-f", "--folder", type=str, help="Folder of notebooks")
    
    parser.add_argument("-p", "--project", type=str, required=True, help="Project Name (Context)")
    parser.add_argument("--nested", action="store_true", help="Search recursively in subfolders")
    parser.add_argument("--force", action="store_true", help="Overwrite existing _explained files")
    
    return parser.parse_args()

async def async_main():
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set in environment or .env file.")
        sys.exit(1)

    args = parse_args()
    explainer = NotebookExplainer()
    processor = BatchProcessor(explainer)

    try:
        if args.notebook:
            path = Path(args.notebook)
            await explainer.process_notebook_async(str(path), args.project)
        
        elif args.folder:
            print(f"➤ Scanning folder: {args.folder} {'(Recursive)' if args.nested else ''}")
            notebooks = processor.scan_notebooks(args.folder, nested=args.nested)
            if not notebooks:
                print("⚠ No .ipynb files found.")
                return
            await processor.process_batch_async(notebooks, args.project, force=args.force)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user.")
        sys.exit(1)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()