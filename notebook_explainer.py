#!/usr/bin/env python3
"""Jupyter Notebook Code Explainer - Batch Processing Edition

Processes Jupyter notebooks and adds AI-generated explanations below each code cell
Also simplifies existing Markdown cells
Supports single file, folder, and nested folder processing with failure recovery
"""

import os
import json
import time
import argparse
import gc
from pathlib import Path
from openai import OpenAI
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
FAILURE_RETRY_DELAY = 5
FAILURE_MAX_RETRIES = 3

# === PROMPTS ===
CODE_EXPLANATION_PROMPT = """You are a patient, enthusiastic programming teacher standing in front of a classroom of curious beginners. Your job is to walk the students through the code below as if you are telling a story together. Speak in natural, connected paragraphs. Never use headings, bullet lists, or the phrase "this code / this snippet". Instead, describe *what the program is trying to achieve* and *how each line contributes*—use gentle analogies (a loop is like repeating a recipe step, a function is a reusable recipe card, etc.). If something is a key concept, explain it in one smooth sentence right where it appears. Keep the whole explanation under 300 words and end with a short tie-back to the larger project.
Project context: {project_name}
Code to walk through:
```python
{code_content}
```
Just write the lecture—nothing else."""

MARKDOWN_REWRITE_PROMPT = """You are a friendly writing coach who turns dense technical text into clear, engaging prose for beginners.
Take the original markdown below and rewrite it in short, warm paragraphs. Use everyday words, break long sentences, and explain any jargon in the same sentence (e.g., "A DataFrame is just a smart table…").
Keep every important fact, but add one extra sentence of helpful context when it would make the idea easier to grasp (e.g., why we need this step, what it prepares us for).
No headings, no bullet lists unless they truly simplify. Use bold or italics only for emphasis. Stay under 250 words.
Project context: {project_name}
Original markdown:
{markdown_content}
Return only the rewritten paragraphs."""

EXPLANATION_PROMPT = CODE_EXPLANATION_PROMPT
MARKDOWN_PROMPT = MARKDOWN_REWRITE_PROMPT


class NotebookExplainer:
    """Process Jupyter notebooks: explain code cells and simplify markdown cells"""

    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
        self.client = OpenAI(api_key=api_key)
        print(f"Initialized Notebook Explainer (Model: {self.model})")

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
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid notebook format: {e}")
        except Exception as e:
            raise Exception(f"Error reading notebook: {e}")

    def get_processable_cells(self, notebook_data: dict) -> list:
        """Extract all code and markdown cells from notebook"""
        cells = notebook_data.get('cells', [])
        processable_cells = []
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type')
            if cell_type in ['code', 'markdown']:
                content = ''.join(cell.get('source', []))
                # Skip empty cells
                if content.strip():
                    processable_cells.append({
                        'index': i,
                        'type': cell_type,
                        'content': content,
                        'execution_count': cell.get('execution_count', None) if cell_type == 'code' else None
                    })
        return processable_cells

    def explain_code(self, code_content: str, project_name: str) -> Optional[str]:
        """Get explanation from OpenAI API for code cells"""
        prompt = EXPLANATION_PROMPT.format(
            project_name=project_name,
            code_content=code_content
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert programming professor who lectures clearly and engagingly."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                if response.choices and response.choices[0].message:
                    explanation = response.choices[0].message.content.strip()
                    return explanation
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise e
        
        return None

    def simplify_markdown(self, markdown_content: str, project_name: str) -> Optional[str]:
        """Get simplified version from OpenAI API for markdown cells"""
        prompt = MARKDOWN_PROMPT.format(
            project_name=project_name,
            markdown_content=markdown_content
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert writing instructor who simplifies text clearly and accessibly."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                if response.choices and response.choices[0].message:
                    simplified = response.choices[0].message.content.strip()
                    return simplified
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise e
        
        return None

    def insert_explanation(self, notebook_data: dict, cell_index: int, explanation: str) -> dict:
        """Insert markdown cell with explanation after the code cell"""
        explanation_cell = {
            "cell_type": "markdown",
            "metadata": {
                "ai_generated": True,
                "explanation": True
            },
            "source": [explanation]
        }
        
        notebook_data['cells'].insert(cell_index + 1, explanation_cell)
        return notebook_data

    def update_markdown_cell(self, notebook_data: dict, cell_index: int, simplified: str) -> dict:
        """Update the existing markdown cell with simplified content"""
        notebook_data['cells'][cell_index]['source'] = [simplified]

        if 'metadata' not in notebook_data['cells'][cell_index]:
            notebook_data['cells'][cell_index]['metadata'] = {}
        
        notebook_data['cells'][cell_index]['metadata'].update({
            "ai_simplified": True
        })
        return notebook_data

    def save_notebook(self, notebook_data: dict, output_path: str) -> bool:
        """Save notebook to disk"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save: {e}")
            return False

    def process_notebook(self, notebook_path: str, project_name: str, output_path: str = None, 
                        show_detailed_progress: bool = True, notebook_number: Optional[int] = None, 
                        total_notebooks: Optional[int] = None) -> Tuple[bool, List[Dict]]:
        """Main processing function - returns (success, failures_list)"""
        
        # Determine output path
        if output_path is None:
            input_path = Path(notebook_path)
            output_path = input_path.parent / f"{input_path.stem}_explained{input_path.suffix}"
            output_path = str(output_path)
        
        # Read notebook
        try:
            notebook_data = self.read_notebook(notebook_path)
        except Exception as e:
            print(f"✗ Failed to read notebook: {e}")
            return False, []
        
        # Get processable cells
        processable_cells = self.get_processable_cells(notebook_data)
        if not processable_cells:
            print(f"⚠ No code or markdown cells found")
            return True, []
        
        # Show header
        if show_detailed_progress:
            if notebook_number and total_notebooks:
                print(f"\n[{notebook_number}/{total_notebooks}] 📓 {Path(notebook_path).name}")
            else:
                print(f"\n📓 Processing: {Path(notebook_path).name}")
            print(f"      Cells: {len([c for c in processable_cells if c['type'] == 'code'])} code, "
                  f"{len([c for c in processable_cells if c['type'] == 'markdown'])} markdown")
        
        # Process each cell
        stats = {
            'success': 0,
            'failed': 0,
            'total': len(processable_cells)
        }
        failures = []
        
        offset = 0
        for i, cell in enumerate(processable_cells, 1):
            cell_type = cell['type']
            
            # Update progress bar
            if show_detailed_progress:
                progress = self._format_progress_bar(i, len(processable_cells), stats['success'], stats['failed'])
                print(f"\r      {progress}", end='', flush=True)
            
            try:
                if cell_type == 'code':
                    explanation = self.explain_code(cell['content'], project_name)
                    if explanation:
                        current_index = cell['index'] + offset
                        notebook_data = self.insert_explanation(notebook_data, current_index, explanation)
                        offset += 1
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                        failures.append({
                            'cell_index': cell['index'],
                            'cell_type': cell_type,
                            'content_preview': cell['content'][:100],
                            'error': 'Failed to get explanation',
                            'timestamp': datetime.now().isoformat()
                        })
                
                elif cell_type == 'markdown':
                    simplified = self.simplify_markdown(cell['content'], project_name)
                    if simplified:
                        current_index = cell['index'] + offset
                        notebook_data = self.update_markdown_cell(notebook_data, current_index, simplified)
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                        failures.append({
                            'cell_index': cell['index'],
                            'cell_type': cell_type,
                            'content_preview': cell['content'][:100],
                            'error': 'Failed to simplify markdown',
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Save after each cell
                self.save_notebook(notebook_data, output_path)
                
                # Small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                stats['failed'] += 1
                failures.append({
                    'cell_index': cell['index'],
                    'cell_type': cell_type,
                    'content_preview': cell['content'][:100],
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Final progress update
        if show_detailed_progress:
            progress = self._format_progress_bar(stats['total'], stats['total'], stats['success'], stats['failed'])
            print(f"\r      {progress}")
        
        # Cleanup memory
        self._cleanup_notebook_memory(notebook_data, processable_cells)
        
        return True, failures

    def _format_progress_bar(self, current: int, total: int, success: int, failed: int, width: int = 20) -> str:
        """Format a progress bar with stats"""
        filled = int(width * current / total)
        bar = '█' * filled + '░' * (width - filled)
        return f"Progress: [{bar}] {current}/{total} | ✓ {success} | ✗ {failed}"

    def _cleanup_notebook_memory(self, *objects):
        """Explicitly cleanup memory after processing a notebook"""
        for obj in objects:
            del obj
        gc.collect()


class BatchProcessor:
    """Handle batch processing of multiple notebooks"""
    
    def __init__(self, explainer: NotebookExplainer):
        self.explainer = explainer
    
    def scan_notebooks(self, folder_path: str, nested: bool = False) -> List[Path]:
        """Scan folder for notebooks, return list of paths"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if nested:
            notebooks = list(folder.rglob('*.ipynb'))
        else:
            notebooks = list(folder.glob('*.ipynb'))
        
        # Filter out checkpoint files AND files that end in '_explained.ipynb'
        notebooks = [
            nb for nb in notebooks 
            if '.ipynb_checkpoints' not in str(nb) 
            and not nb.name.endswith('_explained.ipynb')
        ]
        
        return sorted(notebooks)
    
    def check_already_processed(self, notebook_path: Path) -> bool:
        """Check if notebook has already been processed"""
        explained_path = notebook_path.parent / f"{notebook_path.stem}_explained{notebook_path.suffix}"
        return explained_path.exists()
    
    def display_scan_summary(self, notebooks: List[Path], force: bool = False):
        """Display scan results in a nice format"""
        print("\n╔════════════════════════════════════════════════════════╗")
        print("║       JUPYTER NOTEBOOK EXPLAINER - BATCH MODE         ║")
        print("╚════════════════════════════════════════════════════════╝\n")
        
        # Group by folder
        by_folder = {}
        skipped = []
        to_process = []
        
        for nb in notebooks:
            folder = nb.parent
            if folder not in by_folder:
                by_folder[folder] = []
            by_folder[folder].append(nb)
            
            if self.check_already_processed(nb) and not force:
                skipped.append(nb)
            else:
                to_process.append(nb)
        
        print(f"📂 Found {len(notebooks)} notebook(s) in {len(by_folder)} folder(s)\n")
        
        # Display by folder
        for folder, nbs in sorted(by_folder.items()):
            print(f"  📁 {folder}")
            for nb in nbs:
                if nb in skipped:
                    print(f"     ⊗ {nb.name} (SKIPPED - already processed)")
                else:
                    # Count cells
                    try:
                        with open(nb, 'r') as f:
                            data = json.load(f)
                            cells = data.get('cells', [])
                            code_count = len([c for c in cells if c.get('cell_type') == 'code' and ''.join(c.get('source', [])).strip()])
                            md_count = len([c for c in cells if c.get('cell_type') == 'markdown' and ''.join(c.get('source', [])).strip()])
                        print(f"     ✓ {nb.name} ({code_count} code, {md_count} markdown)")
                    except:
                        print(f"     ✓ {nb.name}")
        
        print(f"\n{'━' * 58}")
        print(f"Total to process: {len(to_process)} notebook(s)")
        if skipped:
            print(f"Skipped: {len(skipped)} (use --force to reprocess)")
        print(f"{'━' * 58}\n")
        
        return to_process
    
    def log_failure(self, notebook_path: Path, failures: List[Dict]):
        """Log failures to JSON file"""
        if not failures:
            return
        
        failure_log = {
            'notebook': str(notebook_path),
            'timestamp': datetime.now().isoformat(),
            'failure_count': len(failures),
            'failures': failures
        }
        
        log_path = notebook_path.parent / f"{notebook_path.stem}_failures.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(failure_log, f, indent=2, ensure_ascii=False)
        
        return log_path
    
    def retry_failures(self, notebook_path: Path, project_name: str) -> List[Dict]:
        """Retry failed cells from failure log"""
        log_path = notebook_path.parent / f"{notebook_path.stem}_failures.json"
        
        if not log_path.exists():
            return []
        
        # Load failure log
        with open(log_path, 'r', encoding='utf-8') as f:
            failure_log = json.load(f)
        
        failures = failure_log.get('failures', [])
        if not failures:
            return []
        
        # Load notebook
        explained_path = notebook_path.parent / f"{notebook_path.stem}_explained{notebook_path.suffix}"
        with open(explained_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        remaining_failures = []
        
        for failure in failures:
            cell_index = failure['cell_index']
            cell_type = failure['cell_type']
            
            print(f"   [Retry] Cell {cell_index} ({cell_type})...", end=' ', flush=True)
            
            success = False
            for attempt in range(FAILURE_MAX_RETRIES):
                try:
                    if cell_type == 'code':
                        # Find the cell
                        cell_content = ''.join(notebook_data['cells'][cell_index].get('source', []))
                        explanation = self.explainer.explain_code(cell_content, project_name)
                        if explanation:
                            notebook_data = self.explainer.insert_explanation(notebook_data, cell_index, explanation)
                            self.explainer.save_notebook(notebook_data, str(explained_path))
                            print("✓")
                            success = True
                            break
                    
                    elif cell_type == 'markdown':
                        cell_content = ''.join(notebook_data['cells'][cell_index].get('source', []))
                        simplified = self.explainer.simplify_markdown(cell_content, project_name)
                        if simplified:
                            notebook_data = self.explainer.update_markdown_cell(notebook_data, cell_index, simplified)
                            self.explainer.save_notebook(notebook_data, str(explained_path))
                            print("✓")
                            success = True
                            break
                except Exception as e:
                    if attempt < FAILURE_MAX_RETRIES - 1:
                        time.sleep(FAILURE_RETRY_DELAY)
            
            if not success:
                print("✗")
                remaining_failures.append(failure)
        
        # Update or delete failure log
        if remaining_failures:
            failure_log['failures'] = remaining_failures
            failure_log['retry_timestamp'] = datetime.now().isoformat()
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(failure_log, f, indent=2, ensure_ascii=False)
        else:
            # All retries succeeded, delete log
            log_path.unlink()
        
        return remaining_failures
    
    def process_batch(self, notebooks: List[Path], project_name: str, force: bool = False) -> Dict:
        """Process multiple notebooks"""
        to_process = []
        
        for nb in notebooks:
            if not self.check_already_processed(nb) or force:
                to_process.append(nb)
        
        if not to_process:
            print("⚠ All notebooks already processed. Use --force to reprocess.\n")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"Starting batch processing...\n")
        print(f"{'━' * 58}\n")
        
        results = {
            'total': len(to_process),
            'success': 0,
            'failed': 0,
            'failure_logs': []
        }
        
        # First pass: process all notebooks
        for i, notebook_path in enumerate(to_process, 1):
            success, failures = self.explainer.process_notebook(
                str(notebook_path),
                project_name,
                output_path=str(notebook_path.parent / f"{notebook_path.stem}_explained{notebook_path.suffix}"),
                show_detailed_progress=True,
                notebook_number=i,
                total_notebooks=len(to_process)
            )
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
            
            # Log failures if any
            if failures:
                log_path = self.log_failure(notebook_path, failures)
                results['failure_logs'].append((notebook_path, log_path))
        
        print(f"\n{'━' * 58}\n")
        
        # Second pass: retry failures
        if results['failure_logs']:
            print(f"RETRY PASS - {len(results['failure_logs'])} notebook(s) with failures\n")
            
            for notebook_path, log_path in results['failure_logs']:
                print(f"📓 {notebook_path.name}")
                remaining = self.retry_failures(notebook_path, project_name)
                
                if not remaining:
                    print(f"   ✅ All failures resolved\n")
                else:
                    print(f"   ⚠ {len(remaining)} cell(s) still failed (logged)\n")
            
            print(f"{'━' * 58}\n")
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain code and simplify markdown in Jupyter notebooks using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    python script.py -n notebook.ipynb -p "ML Project"
  
  Folder (non-recursive):
    python script.py -f ./notebooks -p "Data Analysis"
  
  Nested folders (recursive):
    python script.py -f ./project --nested -p "Research"
  
  Force reprocess:
    python script.py -f ./notebooks --force -p "Updated"
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--notebook", "-n", type=str, help="Path to single .ipynb file")
    mode_group.add_argument("--folder", "-f", type=str, help="Path to folder containing notebooks")
    
    # Required arguments
    parser.add_argument("--project", "-p", type=str, required=True, help="Project name (used in AI context)")
    
    # Optional arguments
    parser.add_argument("--nested", action="store_true", help="Recursively search nested folders (use with --folder)")
    parser.add_argument("--force", action="store_true", help="Reprocess notebooks that already have _explained versions")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for single file mode (default: input_explained.ipynb)")
    parser.add_argument("--model", "-m", type=str, default=MODEL_NAME, help=f"OpenAI model (default: {MODEL_NAME})")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ OPENAI_API_KEY not found!")
        print("\nPlease create a .env file with your OpenAI API key:")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("\nOr set it as an environment variable:")
        print("  export OPENAI_API_KEY=your-api-key-here\n")
        sys.exit(1)
    
    args = parse_args()
    
    # Validate nested flag
    if args.nested and not args.folder:
        print("\n❌ Error: --nested flag requires --folder\n")
        sys.exit(1)
    
    # Create explainer instance
    try:
        explainer = NotebookExplainer(model=args.model)
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}\n")
        sys.exit(1)
    
    # Single file mode
    if args.notebook:
        print("\n" + "═" * 58)
        print("  SINGLE FILE MODE")
        print("═" * 58)
        
        notebook_path = Path(args.notebook)
        
        # Check if already processed
        if BatchProcessor(explainer).check_already_processed(notebook_path) and not args.force:
            print(f"\n⊗ Notebook already processed: {notebook_path.name}")
            print(f"   Found: {notebook_path.stem}_explained{notebook_path.suffix}")
            print(f"   Use --force to reprocess\n")
            sys.exit(0)
        
        success, failures = explainer.process_notebook(
            notebook_path=args.notebook,
            project_name=args.project,
            output_path=args.output,
            show_detailed_progress=True
        )
        
        if success:
            if failures:
                processor = BatchProcessor(explainer)
                log_path = processor.log_failure(notebook_path, failures)
                print(f"\n⚠ Completed with {len(failures)} failure(s)")
                print(f"   Log: {log_path}\n")
            else:
                print(f"\n✅ Processing completed successfully!\n")
        else:
            print(f"\n❌ Processing failed\n")
            sys.exit(1)
    
    # Folder/batch mode
    elif args.folder:
        processor = BatchProcessor(explainer)
        
        try:
            # Scan for notebooks
            notebooks = processor.scan_notebooks(args.folder, nested=args.nested)
            
            if not notebooks:
                print(f"\n⚠ No notebooks found in: {args.folder}\n")
                sys.exit(0)
            
            # Display summary
            to_process = processor.display_scan_summary(notebooks, force=args.force)
            
            if not to_process:
                sys.exit(0)
            
            # Process batch
            results = processor.process_batch(to_process, args.project, force=args.force)
            
            # Final summary
            print("╔════════════════════════════════════════════════════════╗")
            print("║                    FINAL SUMMARY                       ║")
            print("╚════════════════════════════════════════════════════════╝\n")
            print(f"  Total notebooks: {results['total']}")
            print(f"  ✅ Successfully processed: {results['success']}")
            print(f"  ❌ Failed: {results['failed']}")
            
            if results['failure_logs']:
                print(f"\n  ⚠ Failure logs created:")
                for nb_path, log_path in results['failure_logs']:
                    print(f"     - {log_path.name}")
            
            print(f"\n{'═' * 58}\n")
            
        except Exception as e:
            print(f"\n❌ Batch processing failed: {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()