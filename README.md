# Mail Agent (LangChain ReAct)

An intelligent AI agent using the ReAct paradigm to answer queries by dynamically selecting the best source of information, powered by Meta-Llama-3-70B-Instruct.

## Implemented Tools
1. **YouTube Search** (`youtube_search`)
2. **Gmail** (`gmail`)
3. **Local Knowledge Base** (`rag`) - reads your local PDF resume

## Setup Instructions

1. **Activate Environment**
   Open a terminal and activate the virtual environment:
   ```cmd
   .\venv\Scripts\activate
   ```

2. **Add Your Resume**
   Place your PDF resume into this folder and rename it to `resume.pdf`.

3. **Initialize the Knowledge Base**
   Run the ingestion script to create the local FAISS vector database from your resume:
   ```cmd
   python ingest.py
   ```

4. **Run the Agent**
   Start the interactive agent shell:
   ```cmd
   python agent.py
   ```

5. **Test the Agent**
   Use these prompts to test its tools:
   - "Search YouTube for the best videos to learn FastAPI"
   - "Get the subject of my latest 5 emails"
   - "Explain my skills from my resume"
   - "What are transformers in AI?"
