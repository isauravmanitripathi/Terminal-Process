# Terminal Task 


## Notebook Explainer 

This is a Python script that uses the OpenAI API to automatically make Jupyter Notebooks (`.ipynb` files) easier to understand.

It reads a notebook, processes its content, and generates a new, "explained" notebook.

### What It Does

This tool performs two main tasks:

1.  **Explains Code Cells:** For every code cell, it asks an AI to generate a simple, teacher-like explanation. It then inserts this explanation as a new **markdown cell** directly below the code.
2.  **Simplifies Markdown Cells:** For every existing text (markdown) cell, it asks the AI to rewrite the content to be clearer, more engaging, and easier for a beginner to read.

### How It Works

1.  The script takes a notebook file and a "project name" as input.
2.  It loops through every cell in the notebook.
3.  **If a code cell:** It sends the code to the OpenAI API (e.g., `gpt-5-mini`) with a prompt asking it to act like a teacher and explain the code.
4.  **If a markdown cell:** It sends the text to the OpenAI API with a prompt asking it to act like a writing coach and simplify the text.
5.  It saves all the changes (new explanation cells and updated markdown cells) to a new output file (e.g., `your-notebook_explained.ipynb`), leaving your original file untouched.
