#!/usr/bin/env python3
"""Jupyter Notebook Code Explainer - Async Batch Processing with State Tracking

Features:
- Parallel Processing (AsyncIO).
- **State Tracking:** Resumes from where it left off using `processing_tracker.json`.
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
MODEL_NAME = "gpt-5-mini" 
MAX_RETRIES = 3
RETRY_DELAY = 2
CONCURRENT_REQUESTS = 10 

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


class ProgressTracker:
    """Manages state persistence to allow resuming"""
    
    def __init__(self, root_path: Path):
        self.file_path = root_path / "processing_tracker.json"
        self.lock = asyncio.Lock()
        self.data = self._load()

    def _load(self) -> Dict:
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    print(f"✓ Found existing tracker: {self.file_path.name} (Resuming...)")
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Error loading tracker (starting fresh): {e}")
        return {}

    async def save(self):
        """Thread-safe save to disk"""
        async with self.lock:
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠ Failed to write tracker: {e}")

    def get_result(self, notebook_path: str, cell_index: int) -> Optional[Dict]:
        """Check if we already processed this cell"""
        nb_key = str(notebook_path)
        if nb_key in self.data:
            cell_key = str(cell_index)
            if cell_key in self.data[nb_key]['cells']:
                return self.data[nb_key]['cells'][cell_key]
        return None

    async def update_result(self, notebook_path: str, cell_index: int, cell_type: str, output: str, success: bool):
        """Update state for a specific cell"""
        nb_key = str(notebook_path)
        cell_key = str(cell_index)
        
        async with self.lock:
            if nb_key not in self.data:
                self.data[nb_key] = {"status": "in_progress", "cells": {}}
            
            self.data[nb_key]['cells'][cell_key] = {
                "type": cell_type,
                "success": success,
                "output": output,
                "timestamp": datetime.now().isoformat()
            }
        
        # Auto-save on every update to ensure crash recovery
        await self.save()

    async def mark_notebook_complete(self, notebook_path: str):
        nb_key = str(notebook_path)
        async with self.lock:
            if nb_key in self.data:
                self.data[nb_key]['status'] = "completed"
        await self.save()


class NotebookExplainer:
    """Process Jupyter notebooks: explain code cells and simplify markdown cells (Async)"""

    def __init__(self, tracker: ProgressTracker, model: str = MODEL_NAME):
        self.model = model
        self.tracker = tracker
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.client = AsyncOpenAI(api_key=api_key)
        print(f"✓ Initialized Async Notebook Explainer (Model: {self.model})")

    def read_notebook(self, notebook_path: str) -> dict:
        path = Path(notebook_path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error reading notebook: {e}")

    def clean_empty_cells(self, notebook_data: dict) -> Tuple[dict, int]:
        original_cells = notebook_data.get('cells', [])
        cleaned_cells = []
        for cell in original_cells:
            source_content = "".join(cell.get('source', [])).strip()
            if source_content:
                cleaned_cells.append(cell)
        
        removed_count = len(original_cells) - len(cleaned_cells)
        notebook_data['cells'] = cleaned_cells
        return notebook_data, removed_count

    def get_processable_cells(self, notebook_data: dict) -> list:
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
        prompt = EXPLANATION_PROMPT.format(project_name=project_name, code_content=code_content)
        return await self._make_api_call(prompt, "You are an expert technical writer.")

    async def simplify_markdown_async(self, markdown_content: str, project_name: str) -> Optional[str]:
        prompt = MARKDOWN_PROMPT.format(markdown_content=markdown_content)
        return await self._make_api_call(prompt, "You are an expert technical editor.")

    async def _make_api_call(self, prompt: str, system_role: str) -> Optional[str]:
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
                await asyncio.sleep(wait_time)
            except APIError:
                await asyncio.sleep(RETRY_DELAY)
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return None
        return None

    async def process_cell_task(self, semaphore: asyncio.Semaphore, cell: dict, project_name: str, notebook_path: str, progress_callback) -> dict:
        async with semaphore:
            result = {
                'index': cell['index'],
                'type': cell['type'],
                'original_content': cell['content'],
                'success': False,
                'output': None,
                'error': None,
                'cached': False
            }
            
            # 1. CHECK TRACKER FIRST (Resuming)
            cached_data = self.tracker.get_result(notebook_path, cell['index'])
            if cached_data and cached_data.get('success'):
                result['success'] = True
                result['output'] = cached_data['output']
                result['cached'] = True
                progress_callback(True, True) # success, is_cached
                return result

            # 2. PROCESS IF NOT IN TRACKER
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
            
            # 3. UPDATE TRACKER
            await self.tracker.update_result(
                notebook_path, 
                cell['index'], 
                cell['type'], 
                result['output'], 
                result['success']
            )

            progress_callback(result['success'], False)
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
        
        input_path = Path(notebook_path)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_explained{input_path.suffix}"
            output_path = str(output_path)
        
        # 1. Read & Clean
        try:
            notebook_data = self.read_notebook(notebook_path)
            notebook_data, removed_count = self.clean_empty_cells(notebook_data)
        except Exception as e:
            print(f"✗ Failed to read notebook: {e}")
            return False, []
        
        processable_cells = self.get_processable_cells(notebook_data)
        if not processable_cells:
            return True, []
        
        # 2. Display Info
        if show_detailed_progress:
            prefix = f"[{notebook_number}/{total_notebooks}] " if notebook_number else ""
            print(f"\n{prefix}📓 {input_path.name}")
            print(f"   ➤ Cells: {len(processable_cells)} | Cleaned: {removed_count}")

        # 3. Async Execution
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        stats = {'processed': 0, 'success': 0, 'failed': 0, 'cached': 0, 'total': len(processable_cells)}
        
        def update_progress(success, is_cached):
            stats['processed'] += 1
            if success: stats['success'] += 1
            else: stats['failed'] += 1
            if is_cached: stats['cached'] += 1
            
            if show_detailed_progress:
                filled = int(30 * stats['processed'] / stats['total'])
                bar = '█' * filled + '░' * (30 - filled)
                print(f"\r   ➤ Progress: [{bar}] {stats['processed']}/{stats['total']} | Cached: {stats['cached']} | New: {stats['processed'] - stats['cached']}", end='', flush=True)

        tasks = [
            self.process_cell_task(semaphore, cell, project_name, str(input_path), update_progress)
            for cell in processable_cells
        ]
        
        results = await asyncio.gather(*tasks)
        
        if show_detailed_progress:
            print() 

        # 4. Reconstruct & Save
        failures = []
        
        # Apply Markdown (Updates)
        for res in results:
            if not res['success']:
                failures.append({'cell_index': res['index'], 'cell_type': res['type'], 'error': res['error']})
                continue
            
            if res['type'] == 'markdown':
                cell = notebook_data['cells'][res['index']]
                cell['source'] = [res['output']]
                if 'metadata' not in cell: cell['metadata'] = {}
                cell['metadata']['ai_simplified'] = True

        # Apply Code (Insertions - Reverse Order)
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
            notebook_data['cells'].insert(res['index'] + 1, explanation_cell)

        # Mark complete in tracker and save file
        if not failures:
            await self.tracker.mark_notebook_complete(str(input_path))

        if self.save_notebook(notebook_data, output_path):
            print(f"   ➤ Saved to: {Path(output_path).name}")
        
        self._cleanup_notebook_memory(notebook_data, processable_cells, results)
        return True, failures

    def _cleanup_notebook_memory(self, *objects):
        for obj in objects:
            del obj
        gc.collect()


class BatchProcessor:
    def __init__(self, explainer: NotebookExplainer):
        self.explainer = explainer
    
    def scan_notebooks(self, folder_path: str, nested: bool = False) -> List[Path]:
        folder = Path(folder_path)
        if not folder.exists(): raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        glob_pattern = '*.ipynb' if not nested else '**/*.ipynb'
        notebooks = list(folder.glob(glob_pattern)) if not nested else list(folder.rglob(glob_pattern))
        
        filtered = [
            nb for nb in notebooks 
            if '.ipynb_checkpoints' not in str(nb) 
            and not nb.name.lower().endswith('_explained.ipynb')
        ]
        return sorted(filtered)

    async def process_batch_async(self, notebooks: List[Path], project_name: str, force: bool = False):
        # NOTE: With the tracker, we generally don't skip based on file existence alone,
        # but we filter out output files.
        
        print(f"\n╔══════════════════════════════════════════════════════╗")
        print(f"║   Starting Async Batch Processing ({len(notebooks)} files)       ║")
        print(f"╚══════════════════════════════════════════════════════╝")
        
        failure_logs = []
        for i, nb in enumerate(notebooks, 1):
            success, failures = await self.explainer.process_notebook_async(
                str(nb), project_name, notebook_number=i, total_notebooks=len(notebooks)
            )
            
            if failures:
                log_path = self.log_failure(nb, failures)
                failure_logs.append((nb, log_path))
        
        print(f"\n{'━' * 60}")
        print("Batch Complete.")
        if failure_logs:
            print(f"⚠ Failures logged for {len(failure_logs)} notebooks.")

    def log_failure(self, notebook_path: Path, failures: List[Dict]):
        log_path = notebook_path.parent / f"{notebook_path.stem}_failures.json"
        data = {'notebook': str(notebook_path), 'timestamp': datetime.now().isoformat(), 'failures': failures}
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Async Jupyter Notebook Explainer with Tracking")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-n", "--notebook", type=str, help="Single notebook file")
    mode.add_argument("-f", "--folder", type=str, help="Folder of notebooks")
    parser.add_argument("-p", "--project", type=str, required=True, help="Project Name (Context)")
    parser.add_argument("--nested", action="store_true", help="Search recursively in subfolders")
    parser.add_argument("--force", action="store_true", help="Overwrite existing _explained files")
    return parser.parse_args()

async def async_main():
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    args = parse_args()

    # Initialize Tracker based on mode
    if args.folder:
        root_path = Path(args.folder)
    else:
        root_path = Path(args.notebook).parent
    
    tracker = ProgressTracker(root_path)
    explainer = NotebookExplainer(tracker)
    processor = BatchProcessor(explainer)

    try:
        if args.notebook:
            path = Path(args.notebook)
            await explainer.process_notebook_async(str(path), args.project, notebook_number=1, total_notebooks=1)
        
        elif args.folder:
            print(f"➤ Scanning folder: {args.folder} {'(Recursive)' if args.nested else ''}")
            notebooks = processor.scan_notebooks(args.folder, nested=args.nested)
            if not notebooks:
                print("⚠ No .ipynb files found.")
                return
            await processor.process_batch_async(notebooks, args.project, force=args.force)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Paused. Run again to resume exactly where you left off.")
        sys.exit(1)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()