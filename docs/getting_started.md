# Getting Started

## What is agentics?

Agentics is a lightweight, Python-native framework for building structured, agentic workflows over tabular or JSON-based data using Pydantic types and transduction logic. Designed to work seamlessly with large language models (LLMs), Agentics enables users to define input and output schemas as structured types and apply declarative, composable transformations—called transductions—across data collections. It supports asynchronous execution, built-in memory for structured retrieval-augmented generation (RAG), and self-transduction for tasks like data imputation and few-shot learning. With no-code and low-code interfaces, Agentics is ideal for rapidly prototyping intelligent systems that require structured reasoning, flexible memory access, and interpretable outputs.

## Installation

* Clone the repository

  ```shell
    git clone https://github.ibm.com/nl2insights/agentics.git
    cd agentics
  ```

=== "Poetry"

    Install poetry (skip if available)

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Clone and install agentics

    ```bash
    
    poetry install
    poetry shell
    poetry add src/agentics/memory/memory-backend-client
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


### 🎯 Set Environment Variables

Create a `.env` file in the root directory with your environment variables. See `.env.sample` for an example.



The following environment variables are required depending on which components you use:

For WatsonX AI Inference:

- `WATSONX_APIKEY` - WatsonX API key

- `MODEL`  - watsonx/meta-llama/llama-3-3-70b-instruct (or alternative supporting function call)

Configuration folder for memory:

- `AGENT_PERSISTANCE_PATH` - Path to a folder in the local file system where memory will be stored (will create a new empty folder on first use)


### 🎯 Memory

if you need to use memory, you should start the memory backend first. Make sure you have set your AGENT_PERSISTANCE_PATH variable in the .env file to a local folder before you start the server. 

```bash
sh cmd/start_memory_backend.sh
```
Note: You might have already done it as part of the installation procedure, however, you'll need to restart the server any time you restart your machine. 
 
You can turn off the server with this script

```bash
sh cmd/stop_backends.sh
```

To regenerate memory apis
```bash
cd src/agentics/memory/ 
openapi-python-client generate --url http://0.0.0.0:7816/openapi.json --overwrite
poetry update memory-backend-client
```

### 🎯 Test Installation

To test agentics youl'll need to activate the memory server first, then run the tests.

```bash
sh cmd/start_memory_backend.sh
poetry run pytest
```

You are all set!!!!
You can now familiarize with the framework playing with the examples provided in the src/agentics/examples. Enjoy.

```bash
python src/agentics/examples/information_extraction.py
```

## 🎯 Basic Example

This example demonstrates how to use Agentics to transduce a list of natural language prompts into structured answers, using `pydantic` for defining the output schema.

```python
from agentics import Agentics as AG
from pydantic import BaseModel
import asyncio

# Define the structured output schema
class Answer(BaseModel):
    answer: str
    justification: str

async def main():
    # Initialize the Agentics pipeline with the output type
    answers = AG.from_pydantic(Answer)

    # Submit a batch of input questions
    answers = answers << [
        "How many states are in the US?",
        "Who is the greatest philosopher of all time?"
    ]

    # Execute the transduction
    answers = await answers

    # Print structured results
    print(answers.states)

# Run the async main function
asyncio.run(main())
```
This will generate the following answer

```
[Answer(answer='50', justification='The United States is divided into 50 states, which are represented in the US Congress and have their own governments. This number has remained constant since Hawaii became the 50th state in 1959.'), Answer(answer='Aristotle', justification="Aristotle is widely regarded as one of the greatest philosophers of all time, and his works have had a profound impact on Western philosophy. He made significant contributions to various fields, including metaphysics, ethics, politics, and biology. His ideas and writings have influenced many prominent philosophers throughout history, including Plato, Immanuel Kant, and Friedrich Nietzsche. Aristotle's philosophical works, such as 'Nicomachean Ethics' and 'Metaphysics,' are still widely studied and debated today, and his concepts, like the 'four causes' and 'hylomorphism,' remain fundamental to philosophical discourse. Furthermore, Aristotle's philosophical method, which emphasizes observation, experience, and reasoning, has shaped the development of science and philosophy for centuries. Overall, Aristotle's profound influence, enduring relevance, and intellectual breadth make a strong case for him being considered the greatest philosopher of all time.")]
```

## 🎯 Agentics Playground

You can play with Agentics using the streamlit app.

```bash
sh cmd/start_playground.sh
```

Once started, the streamlit server will be available at [http://localhost:8501](http://localhost:8501)



## Documentation

This documentation page is written using Mkdocs. 
You can start the server to visualize this interactively.
```bash
mkdocs serve
```
After started, documentation will be available here [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
