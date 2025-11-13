#!/usr/bin/env python3
"""Jupyter Notebook Code Explainer

Processes a Jupyter notebook and adds AI-generated explanations below each code cell
Also simplifies existing Markdown cells
"""

import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-5-mini"  # Using a cost-effective model
MAX_RETRIES = 3
RETRY_DELAY = 2

# === NEW IMPROVED PROMPTS (only change) ===
CODE_EXPLANATION_PROMPT = """You are a patient, enthusiastic programming teacher standing in front of a classroom of curious beginners. Your job is to walk the students through the code below as if you are telling a story together. Speak in natural, connected paragraphs. Never use headings, bullet lists, or the phrase “this code / this snippet”. Instead, describe *what the program is trying to achieve* and *how each line contributes*—use gentle analogies (a loop is like repeating a recipe step, a function is a reusable recipe card, etc.). If something is a key concept, explain it in one smooth sentence right where it appears. Keep the whole explanation under 300 words and end with a short tie-back to the larger project.
Project context: {project_name}
Code to walk through:
```python
{code_content}
```
Just write the lecture—nothing else."""

MARKDOWN_REWRITE_PROMPT = """You are a friendly writing coach who turns dense technical text into clear, engaging prose for beginners.
Take the original markdown below and rewrite it in short, warm paragraphs. Use everyday words, break long sentences, and explain any jargon in the same sentence (e.g., “A DataFrame is just a smart table…”).
Keep every important fact, but add one extra sentence of helpful context when it would make the idea easier to grasp (e.g., why we need this step, what it prepares us for).
No headings, no bullet lists unless they truly simplify. Use bold or italics only for emphasis. Stay under 250 words.
Project context: {project_name}
Original markdown:
{markdown_content}
Return only the rewritten paragraphs."""

# Use the new prompts
EXPLANATION_PROMPT = CODE_EXPLANATION_PROMPT
MARKDOWN_PROMPT = MARKDOWN_REWRITE_PROMPT

class NotebookExplainer:
    """Process Jupyter notebooks: explain code cells and simplify markdown cells"""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
        self.client = OpenAI(api_key=api_key)
        print(f"Initialized Notebook Explainer")
        print(f"  Model: {self.model}\n")

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
            print(f"Loaded notebook: {path.name}")
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
        print(f"Found {len([c for c in processable_cells if c['type'] == 'code'])} code cells and {len([c for c in processable_cells if c['type'] == 'markdown'])} markdown cells to process\n")
        return processable_cells

    def explain_code(self, code_content: str, project_name: str) -> str:
        """Get explanation from OpenAI API for code cells"""
        # Format prompt
        prompt = EXPLANATION_PROMPT.format(
            project_name=project_name,
            code_content=code_content
        )
        
        # Try with retries
        for attempt in range(MAX_RETRIES):
            try:
                print(f"    Requesting explanation (attempt {attempt + 1}/{MAX_RETRIES})...", end=" ")
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
                    print("Done")
                    return explanation
                else:
                    print("Empty response")
            except Exception as e:
                print(f"Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"    Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
        
        # If all retries fail, return None
        return None

    def simplify_markdown(self, markdown_content: str, project_name: str) -> str:
        """Get simplified version from OpenAI API for markdown cells"""
        # Format prompt
        prompt = MARKDOWN_PROMPT.format(
            project_name=project_name,
            markdown_content=markdown_content
        )
        
        # Try with retries
        for attempt in range(MAX_RETRIES):
            try:
                print(f"    Requesting simplification (attempt {attempt + 1}/{MAX_RETRIES})...", end=" ")
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
                    print("Done")
                    return simplified
                else:
                    print("Empty response")
            except Exception as e:
                print(f"Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"    Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
        
        # If all retries fail, return None
        return None

    def insert_explanation(self, notebook_data: dict, cell_index: int, explanation: str) -> dict:
        """Insert markdown cell with explanation after the code cell"""
        # Create new markdown cell with plain paragraphs
        explanation_cell = {
            "cell_type": "markdown",
            "metadata": {
                "ai_generated": True,
                "explanation": True
            },
            "source": [
                explanation  # Just the raw explanation as paragraphs—no headers or lines
            ]
        }
        
        # Insert after the code cell
        notebook_data['cells'].insert(cell_index + 1, explanation_cell)
        return notebook_data

    def update_markdown_cell(self, notebook_data: dict, cell_index: int, simplified: str) -> dict:
        """Update the existing markdown cell with simplified content"""
        # Update the source of the existing markdown cell
        notebook_data['cells'][cell_index]['source'] = [simplified]

        # Add metadata if not present
        if 'metadata' not in notebook_data['cells'][cell_index]:
            notebook_data['cells'][cell_index]['metadata'] = {}
        
        # Update metadata to mark as simplified
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
            print(f"    Failed to save: {e}")
            return False

    def process_notebook(self, notebook_path: str, project_name: str, output_path: str = None):
        """Main processing function"""
        print("="*70)
        print(f"JUPYTER NOTEBOOK EXPLAINER & SIMPLIFIER")
        print("="*70)
        print(f"\nProject: {project_name}")
        print(f"Notebook: {notebook_path}\n")
        
        # Read notebook
        try:
            notebook_data = self.read_notebook(notebook_path)
        except Exception as e:
            print(f"Failed to read notebook: {e}")
            return False
        
        # Get processable cells
        processable_cells = self.get_processable_cells(notebook_data)
        if not processable_cells:
            print("No code or markdown cells found in notebook")
            return False
        
        # Determine output path upfront
        if output_path is None:
            input_path = Path(notebook_path)
            output_path = input_path.parent / f"{input_path.stem}_explained{input_path.suffix}"
            output_path = str(output_path)  # Ensure it's a string
        
        print(f"Output will be saved to: {output_path}")
        print(f"Notebook will be saved after EACH cell is processed\n")
        
        # Process each cell
        stats = {
            'code_total': len([c for c in processable_cells if c['type'] == 'code']),
            'code_success': 0,
            'code_failed': 0,
            'markdown_total': len([c for c in processable_cells if c['type'] == 'markdown']),
            'markdown_success': 0,
            'markdown_failed': 0
        }
        
        print("="*70)
        print("PROCESSING CELLS (LIVE SAVING ENABLED)")
        print("="*70 + "\n")
        
        # We need to track offset only for insertions (code explanations)
        offset = 0
        for i, cell in enumerate(processable_cells, 1):
            cell_type = cell['type']
            exec_count = cell['execution_count'] or 'N/A' if cell_type == 'code' else ''
            print(f"[{i}/{len(processable_cells)}] Processing {cell_type.upper()} cell {exec_count}:")
            
            # Show content preview (first 100 chars)
            content_preview = cell['content'][:100].replace('\n', ' ')
            if len(cell['content']) > 100:
                content_preview += "..."
            print(f"    Content: {content_preview}")
            
            if cell_type == 'code':
                # Get explanation
                explanation = self.explain_code(cell['content'], project_name)
                if explanation:
                    # Insert explanation after the current cell (with offset)
                    current_index = cell['index'] + offset
                    notebook_data = self.insert_explanation(notebook_data, current_index, explanation)
                    # Increase offset for next iteration (we added one cell)
                    offset += 1
                    stats['code_success'] += 1
                    print(f"    Explanation added")
                else:
                    stats['code_failed'] += 1
                    print(f"    Failed to get explanation")
            
            elif cell_type == 'markdown':
                # Get simplified version
                simplified = self.simplify_markdown(cell['content'], project_name)
                if simplified:
                    # Update the existing markdown cell (no offset change)
                    current_index = cell['index'] + offset  # Offset from previous insertions
                    notebook_data = self.update_markdown_cell(notebook_data, current_index, simplified)
                    stats['markdown_success'] += 1
                    print(f"    Markdown simplified")
                else:
                    stats['markdown_failed'] += 1
                    print(f"    Failed to simplify markdown")
            
            # SAVE IMMEDIATELY AFTER EACH CELL
            print(f"    Saving notebook...", end=" ")
            if self.save_notebook(notebook_data, output_path):
                print("Saved!")
                print(f"    You can open the notebook now to verify: {output_path}\n")
            else:
                print("Save failed\n")
            
            # Small delay to avoid rate limits
            time.sleep(1)

        # Final save confirmation
        print("="*70)
        print(f"ALL CELLS PROCESSED - NOTEBOOK SAVED")
        print("="*70)
        print(f"\nFinal Output: {output_path}")
        print(f"You can verify all explanations and simplifications in the notebook now!")
        
        # Final report
        self.print_final_report(stats, output_path)
        return True

    def print_final_report(self, stats: dict, output_path: str):
        """Print final processing report"""
        print(f"\n{'='*70}")
        print("FINAL REPORT")
        print("="*70)
        print(f"\nSTATISTICS:")
        print(f"  Code cells:")
        print(f"     Total: {stats['code_total']}")
        print(f"     Successfully explained: {stats['code_success']}")
        print(f"     Failed: {stats['code_failed']}")
        code_rate = (stats['code_success'] / stats['code_total'] * 100) if stats['code_total'] > 0 else 0
        print(f"     Success rate: {code_rate:.1f}%")
        
        print(f"  Markdown cells:")
        print(f"     Total: {stats['markdown_total']}")
        print(f"     Successfully simplified: {stats['markdown_success']}")
        print(f"     Failed: {stats['markdown_failed']}")
        md_rate = (stats['markdown_success'] / stats['markdown_total'] * 100) if stats['markdown_total'] > 0 else 0
        print(f"     Success rate: {md_rate:.1f}%")
        
        print(f"\nNEXT STEPS:")
        print(f"   1. Open the explained notebook: {output_path}")
        print(f"   2. Review the AI-generated explanations and simplifications")
        print(f"   3. Edit or refine as needed")
        print(f"   4. Use for learning or documentation!")
        print("\n" + "="*70)

def parse_args():
    parser = argparse.ArgumentParser(description="Explain code and simplify markdown in Jupyter notebooks using AI")
    parser.add_argument("--notebook", "-n", type=str, required=True, help="Path to input .ipynb file")
    parser.add_argument("--project", "-p", type=str, required=True, help="Project name (used in AI context)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path (default: input_explained.ipynb)")
    parser.add_argument("--model", "-m", type=str, default=MODEL_NAME, help=f"OpenAI model (default: {MODEL_NAME})")
    return parser.parse_args()

def main():
    """Main entry point"""
    print("\n" + " " * 20)
    print("JUPYTER NOTEBOOK EXPLAINER & SIMPLIFIER")
    print("Powered by OpenAI GPT")
    print(" " * 20)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\nOPENAI_API_KEY not found!")
        print("\nPlease create a .env file with your OpenAI API key:")
        print("  OPENAI_API_KEY=your-api-key-here")
        print("\nOr set it as an environment variable:")
        print("  export OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
        
    args = parse_args()

    # Create explainer instance
    explainer = NotebookExplainer(model=args.model)

    # Process notebook
    success = explainer.process_notebook(
        notebook_path=args.notebook,
        project_name=args.project,
        output_path=args.output
    )
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()