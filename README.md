# Neural_IR_Expansion: Information Retrieval exam
This project explores **query expansion techniques** using LLMs in the context of **neural information retrieval**. We evaluate a BM25 baseline and a fine-tuned cross-encoder (MiniLM), comparing standard queries to expanded ones generated using a large language model.

## 🚀 How to Run the Project
### 1. uv initialization
We used uv as the Python package manager to ensure consistent dependency resolution and environment setup across different systems.
Install uv:
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Initialize uv and install dependencies
```
$ uv init example
Initialized project `example` at `/home/user/example`

$ cd example

$ uv add ruff
Creating virtual environment at: .venv
Resolved 2 packages in 170ms
   Built example @ file:///home/user/example
Prepared 2 packages in 627ms
Installed 2 packages in 1ms
 + example==0.1.0 (from file:///home/user/example)
 + ruff==0.5.0

$ uv run ruff check
All checks passed!

$ uv lock
Resolved 2 packages in 0.33ms

$ uv sync
Resolved 2 packages in 0.70ms
Audited 1 package in 0.02ms
```
### 3. Run the experiments
Now everything should be set. You can try to run the main.py file, which initiates the IR ranking of the datasets test.csv, novice_expansion.csv, expert_expansion.csv and saves the results in results/results_new.txt.
```
python3 .\src\main.py
```
