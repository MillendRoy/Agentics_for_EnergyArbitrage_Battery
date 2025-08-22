# Getting Started

## What is agentics?

Agentics is a lightweight, Python-native framework for building structured, agentic workflows over tabular or JSON-based data using Pydantic types and transduction logic. Designed to work seamlessly with large language models (LLMs), Agentics enables users to define input and output schemas as structured types and apply declarative, composable transformations—called transductions—across data collections. It supports asynchronous execution, built-in memory for structured retrieval-augmented generation (RAG), and self-transduction for tasks like data imputation and few-shot learning. With no-code and low-code interfaces, Agentics is ideal for rapidly prototyping intelligent systems that require structured reasoning, flexible memory access, and interpretable outputs.

## Installation

* Clone the repository

  ```shell
    git clone git@github.com:IBM/agentics.git
    cd agentics
  ```

### 🎯 Set Environment Variables

Create a `.env` file in the root directory with your environment variables. See `.env.sample` for an example.



The following environment variables are required depending on which components you use:

For WatsonX AI Inference:

- `WATSONX_APIKEY` - WatsonX API key

- `MODEL`  - watsonx/meta-llama/llama-3-3-70b-instruct (or alternative supporting function call)

Configuration folder for memory:

- `AGENT_PERSISTANCE_PATH` - Path to a folder in the local file system where memory will be stored (will create a new empty folder on first use)

For VLLM Inference:

- Set up your local instance of VLLM 
- `VLLM_URL` - <http://base_url:PORT/v1>
- `VLLM_MODEL_ID` - Your model id (e.g. "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct" )


## 🎯 Basic Example

This example demonstrates how to use Agentics to transduce a list of natural language prompts into structured answers, using `pydantic` for defining the output schema.

```python
import asyncio
from pydantic import BaseModel
from agentics import Agentics as AG
from typing import Optional

class Answer(BaseModel):
    answer: Optional[str] = None
    justification: Optional[str] = None
    confidence: Optional[float] = None

async def main():
    input_questions = [
        "What is the capital of Italy?",
        "What is the best F1 team in history?",
        "List games inspiring suicide",
    ]

    answers = await (AG(atype=Answer, 
                        llm= watsonx_crewai_llm,
                        instructions="""Provide an Answer for the following input text 
                        only if it contains an appropriate question that do not contain
                        violent or adult language """
                        ) << input_questions)

    print(answers.pretty_print())

asyncio.run(main())
```
This will generate the following answers

```
answer: Rome
justification: The capital of Italy is a well-known fact that can be found in various
  geographical and educational sources.
confidence: 1.0

answer: null
justification: The input text does not contain a question, it appears to be a statement
  about working with Agentics for the first time.
confidence: 1.0

answer: null
justification: The input text contains a question that may be related to sensitive
  or potentially disturbing topics, but it does not contain violent or adult language.
  However, providing a list of videogames that inspire suicide may be harmful or triggering
  for some individuals. Therefore, it is not possible to provide a logical and safe
  answer to this question.
confidence: 0.0
```



## Documentation

This documentation page is written using Mkdocs. 
You can start the server to visualize this interactively.
```bash
mkdocs serve
```
After started, documentation will be available here [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Installation details

=== "Poetry"

    Install poetry (skip if available)

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Clone and install agentics

    ```bash
    
    poetry install
    poetry shell
    ```

=== "Python"

    > Ensure you have Python 3.11+ 🚨.
    >
    > ```shell
    > python --version
    > ```

    * Create a virtual environment with Python's built in `venv` module. In linux, this 
    package may be required to be installed with the Operating System package manager.
        ```shell
        python -m venv .venv
        ```

    * Activate the virtual environment

    ### Bash/Zsh

    `source .venv/bin/activate`

    ### Fish

    `source .venv/bin/activate.fish`

    ### VSCode 

    Press `F1` key and start typing `> Select python` and select `Select Python Interpreter`

    * Install the package
        ```bash
        python -m pip install ./agentics
        ```
    

=== "uv"

    * Ensure `uv` is installed.
    ```bash
    command -v uv >/dev/null &&  curl -LsSf https://astral.sh/uv/install.sh | sh
    # It's recommended to restart the shell afterwards
    exec $SHELL
    ```
    * `uv venv --python 3.11`
    * `uv pip install ./agentics` or `uv add ./agentics` (recommended)
  

=== "uvx 🏃🏽"

    > This is a way to run agentics temporarily or quick tests

    * Ensure `uv` is installed.
    ```bash
    command -v uv >/dev/null &&  curl -LsSf https://astral.sh/uv/install.sh | sh
    # It's recommended to restart the shell afterwards
    exec $SHELL
    ```
    * uvx --verbose --from ./agentics ipython


=== "Conda"

    1. Create a conda environment:
       ```bash
       conda create -n agentics python=3.11
       ```
       In this example the name of the environment is `agetnics` but you can change
       it to your personal preference.


    2. Activate the environment
        ```bash
        conda activate agentics
        ```
    3. Install `agentics` from a folder or git reference
        ```bash
        pip install ./agentics
        ```
