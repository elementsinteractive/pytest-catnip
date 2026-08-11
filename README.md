# pytest-catnip

<img alt="lgtm-logo" width="150" src="https://raw.githubusercontent.com/elementsinteractive/pytest-catnip/main/assets/logo-small.png">


<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=yellow)

**pytest-catnip** is a pytest plugin for integration-testing [pipecat](https://github.com/pipecat-ai/pipecat) voicebots declaratively, using plain YAML files.

It sits between unit tests and full end-to-end tests: it skips STT and TTS entirely and drives the bot's LLM pipeline directly — letting you verify tool calls, bot replies, and flow-state transitions without any audio infrastructure.

```
Unit tests ──► pytest-catnip ──► Full E2E
               (no STT/TTS,
                real LLM logic)
```

---

**Table of Contents**

- [Why pytest-catnip?](#why-pytest-catnip)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Configure pytest](#1-configure-pytest)
  - [2. Provide the pipeline fixture](#2-provide-the-pipeline-fixture)
  - [3. Write your first test case](#3-write-your-first-test-case)
  - [4. Run](#4-run)
- [Test Case Reference](#test-case-reference)
  - [Top-level fields](#top-level-fields)
  - [Phase fields](#phase-fields)
  - [Asserting tool call arguments](#asserting-tool-call-arguments)
  - [Mocking and Setup (Custom Fixtures)](#mocking-and-setup-custom-fixtures)
- [Fixtures Reference](#fixtures-reference)
  - [`catnip_pipeline`](#catnip_pipeline)
  - [`autoconfirm_response`](#autoconfirm_response)
  - [`is_confirmation_question`](#is_confirmation_question)
  - [`llm_judge_function`](#llm_judge_function)
- [pipecat-flows Integration](#pipecat-flows-integration)
- [Reliability Testing](#reliability-testing)
- [CLI Options & ini Settings](#cli-options--ini-settings)
- [Debugging Failures](#debugging-failures)
- [Contributing](#contributing)

---

## Why pytest-catnip?

Pipecat voicebots are pipelines of frame processors. Full end-to-end tests are slow and fragile — they need microphones, speakers, telephony infrastructure, and a live WebSocket connection. Pure unit tests on individual processors miss the integration between the LLM, tool calls, context aggregation, and flow management.

pytest-catnip fills that gap:

| | Unit tests | **pytest-catnip** | Full E2E |
|---|---|---|---|
| Real LLM calls | No | **Yes** | Yes |
| STT / TTS | No | **No** | Yes |
| Audio / telephony | No | **No** | Yes |
| Tool call assertions | Hard | **Easy** | Hard |
| Flow state assertions | No | **Yes** | No |
| Speed | Fast | **Medium** | Slow |

Tests are written as YAML files — no Python boilerplate for each scenario. The plugin auto-discovers them, generates real pytest test items, and integrates fully with pytest's output, filtering, and coverage.

---

## Installation

```sh
pip install pytest-catnip
```

**Requirements:** Python 3.13+, pipecat-ai ≥ 1.3.0

---

## Quick Start

### 1. Configure pytest

Add to your `pyproject.toml` (or `pytest.ini`):

```toml
[tool.pytest.ini_options]
catnip_cases_directory = "tests/cases"   # where your YAML files live (default)
catnip_show_conversation = true          # print conversation logs
```

### 2. Provide the pipeline fixture

pytest-catnip discovers your bot through a `catnip_pipeline` fixture that you define in `conftest.py`. The fixture must return a **factory callable** — it is called fresh for every test run.

> **Important:** `CatnipTurnTracker` must be in the pipeline. Do **not** add STT or TTS processors.

```python
# tests/conftest.py
import os
from collections.abc import Callable

import pytest
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai import OpenAILLMService
from pytest_catnip import CatnipTurnTracker


@pytest.fixture
def catnip_pipeline() -> Callable[[], Pipeline]:
    def factory() -> Pipeline:
        context = LLMContext(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep replies short."},
            ]
        )
        llm = OpenAILLMService(
            api_key=os.environ.get("OPENAI_API_KEY"), 
            model="gpt-4o-mini"
        )
        user_agg, assistant_agg = LLMContextAggregatorPair(context)
        turn_tracker = CatnipTurnTracker()
        return Pipeline([user_agg, llm, turn_tracker, assistant_agg])

    return factory
```

### 3. Write your first test case

Create `tests/cases/test_greeting.yaml`:

```yaml
name: greeting
description: Bot can answer a basic arithmetic question

phases:
  - send: "Hello! What is 2 + 2?"
    expect_reply_contains:
      - "4"

  - send: "And what is 3 + 5?"
    expect_reply_contains:
      - "8"
```

Files must be named `test_*.yaml` and live inside `catnip_cases_directory`.

### 4. Run

```sh
pytest tests/cases/
```

Each YAML file becomes a single pytest test item, fully integrated with pytest's `-k`, `-x`, `--tb`, and all other standard flags.

---

## Test Case Reference

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Test name. Used as the pytest test ID. |
| `description` | `str` | `""` | Human-readable description. Shown in output. |
| `skip` | `bool` | `false` | Skip this test case. |
| `markers` | `list[str]` | `[]` | List of pytest markers to apply to the test case. |
| `fixtures` | `list[str]` | `[]` | List of pytest fixtures to execute *before* the test starts. Useful for mocking external APIs or setting up state. |
| `max_auto_confirm` | `int` | `0` | Max times the harness auto-answers confirmation questions before failing. See [auto-confirm](#autoconfirm_response). |
| `turn_timeout` | `int` | `15` | Seconds to wait for each bot reply before timing out. |
| `phases` | `list` | required | Ordered list of conversation turns. |
| `reliability` | `object` | `null` | Reliability block — see [Reliability Testing](#reliability-testing). |
| `expect_flow_nodes` | `list[str] \| null` | `null` | Expected pipecat-flow node path. Requires [flows integration](#pipecat-flows-integration)|

### Phase fields

Each entry in `phases` represents one user message and the assertions to run against the bot's response.

| Field | Type | Default | Description |
|---|---|---|---|
| `send` | `str` | required | The user utterance to inject into the pipeline. |
| `expect_reply_contains` | `list[str]` | `[]` | Every string in this list must appear (case-insensitive regex) in the bot's reply. |
| `expect_llm_judge` | `list[str] \| null` | `null` | List of natural language expectations evaluated dynamically by an LLM. See [`llm_judge_function`](#llm_judge_function). |
| `expect_tools` | `list \| null` | `null` | Exact set of tool/function names the LLM must call during this turn. Entries can be plain names or objects that also assert the call **arguments** — see [below](#asserting-tool-call-arguments). `null` skips the assertion. |
| `expect_flow_state` | `str \| null` | `null` | Expected pipecat-flows node name after this phase. Requires [flows integration](#pipecat-flows-integration). |

**Example:**

```yaml
name: book-appointment
description: User books a dental appointment
markers: [integration]
skip: false

phases:
  - send: "I'd like to book a cleaning for next Tuesday at 10am."
    expect_tools:
      - check_availability
    expect_reply_contains:
      - "Tuesday"
    expect_llm_judge:
      - "The bot politely confirms the time."
      - "The bot does not ask for insurance details yet."
    expect_flow_state: appointment_confirmed
```

#### Asserting tool call arguments

Each entry in `expect_tools` is either a plain tool name (string) or an object
with a `name` and an optional `args` mapping. Use the object form to also assert
the **arguments** a tool was called with:

```yaml
phases:
  - send: "Book a cleaning for next Tuesday at 10am."
    expect_tools:
      - name: check_availability
        args:
          service: cleaning   # only the listed keys are checked (partial match)
          time: "10:00"
      - confirm_booking        # string shorthand: name only, no arg checks
    expect_reply_contains:
      - "confirmed"
```

Semantics:

- The set of tool names called must **exactly** match the set of expected names.
- For entries with `args`, the listed keys are matched as a **subset** of the
  actual call arguments — keys you don't list are ignored, and each listed value
  must compare equal to the actual argument.

#### Mocking and Setup (Custom Fixtures)

If your test requires specific database states or mocked tool responses, you can inject standard pytest fixtures defined in your `conftest.py` directly into the YAML using the `fixtures` key.

These fixtures will execute **before** your pipeline factory runs, ensuring your mocks are actively patching the environment before the test begins.

```yaml
name: check-weather
fixtures: [mock_weather_api]  # Injects the fixture from conftest.py

phases:
  - send: "What is the weather like in Barcelona today?"
    expect_tools:
      - check_weather
    expect_reply_contains:
      - "72"
      - "Sunny"
```

---

## Fixtures Reference

### `catnip_pipeline`

**Must be overridden in your `conftest.py`.** Returns a zero-argument factory callable that builds a fresh pipeline (and optionally a `CatnipFlowBundle`) on every call.

The factory is called once per test run, including each reliability run, so every run gets completely fresh, stateful pipecat objects.

### `autoconfirm_response`

When the bot asks a confirmation question (e.g. *"Are you sure?"*), the harness can auto-answer it rather than failing the test. This fixture controls what answer is sent.

**Default:** `"Yes."`

### `is_confirmation_question`

Predicate used to detect whether the bot's reply is a confirmation question. If it returns `True`, the harness sends `autoconfirm_response` instead of advancing to the next phase assertion.

**Default:** any reply that ends with `?`.

### `llm_judge_function`

Used to power the `expect_llm_judge` phase assertion. By default, attempting to use `expect_llm_judge` without overriding this fixture will raise a `NotImplementedError`.

Override this in your `conftest.py` to connect an LLM client. The fixture must return an async callable that takes `(actual_reply: str, expected_judge_prompt: str)` and returns a tuple of `(passed: bool, reason: str)`.

`pytest-catnip` provides several presets to use as a judge:
  - OpenAI judge (install with extra `pytest-catnip[judge-openai]`).
  - Gemini judge (install with extra `pytest-catnip[judge-gemini]`).
  - Anthropic judge (install with extra `pytest-catnip[judge-anthropic]`).


Use it like so:

```python
import pytest
from pytest_catnip.judges import openai_judge

@pytest.fixture(scope="session")
def llm_judge_function():
    return openai_judge(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
```

If you want to customize the judge instead of using one of the given presets, you can create your own judge like in the example below:

**Example using Gemini GenAI SDK and Structured Outputs:**

```python
import os
from collections.abc import Awaitable, Callable
from typing import Annotated

import pytest
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

class JudgeResult(BaseModel):
    passed: Annotated[bool, Field(description="True if the bot reply meets the expectation, False otherwise.")]
    reason: Annotated[str, Field(description="A short explanation of why it passed or failed.")]

@pytest.fixture(scope="session")
def llm_judge_function() -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    async def judge(actual_reply: str, expected_judge_prompt: str) -> tuple[bool, str]:
        prompt = (
            f"You are a strict QA testing judge.\n"
            f"Expectation: {expected_judge_prompt}\n"
            f"Actual Bot Reply: {actual_reply}\n\n"
            f"Evaluate if the bot reply satisfies the expectation."
        )

        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeResult,
                temperature=0.0,
            ),
        )
        result = JudgeResult.model_validate_json(response.text)
        return result.passed, result.reason

    return judge
```

---

## pipecat-flows Integration

If your bot uses [pipecat-flows](https://github.com/pipecat-ai/pipecat-flows) for multi-node conversation management, pytest-catnip can assert:

- That the flow transitions to the expected node after each phase using `expect_flow_state`.
- That the whole flow node path matches expectations using `expect_flow_nodes`.

You can use `expect_flow_state` when you are interested in each phase's node, while `expect_flow_nodes` is for asserting
a specific path for the whole test case, without actually caring about when those node transitions happen.

Instead of returning a plain `Pipeline`, your factory returns a `CatnipFlowBundle`:

```python
import os
from collections.abc import Callable

import pytest
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService
from pipecat.flows import FlowManager, FlowsFunctionSchema
from pipecat.flows.types import FlowArgs, NodeConfig
from pytest_catnip import CatnipTurnTracker
from pytest_catnip.flows import CatnipFlowBundle, CatnipFlowTracker


def _farewell_node() -> NodeConfig:
    return NodeConfig(
        name="farewell",
        task_messages=[{"role": "system", "content": "Wish the user farewell."}],
        functions=[],
    )


def _greeting_node() -> NodeConfig:
    async def handle_say_farewell(args: FlowArgs, fm: FlowManager) -> tuple[None, NodeConfig]:
        return None, _farewell_node()

    return NodeConfig(
        name="greeting",
        respond_immediately=False,
        task_messages=[{"role": "system", "content": "You are a helpful assistant."}],
        functions=[
            FlowsFunctionSchema(
                name="say_farewell",
                description="Call when the user says goodbye.",
                properties={},
                required=[],
                handler=handle_say_farewell,
            )
        ],
    )


@pytest.fixture
def catnip_pipeline() -> Callable[[], CatnipFlowBundle]:
    def factory() -> CatnipFlowBundle:
        context = LLMContext(messages=[{"role": "system", "content": "You are a helpful assistant."}])
        llm = GoogleLLMService(api_key=os.environ.get("GOOGLE_API_KEY"), settings=GoogleLLMService.Settings(model="gemini-2.5-flash"))
        ctx_agg = LLMContextAggregatorPair(context)
        turn_tracker = CatnipTurnTracker()
        pipeline = Pipeline([ctx_agg.user(), llm, turn_tracker, ctx_agg.assistant()])

        async def init_flow(worker: PipelineWorker) -> CatnipFlowTracker:
            tracker = CatnipFlowTracker(worker=worker, llm=llm, context_aggregator=ctx_agg)
            await tracker.initialize(_greeting_node())
            return tracker

        return CatnipFlowBundle(pipeline=pipeline, init_flow=init_flow)

    return factory
```

Then in your YAML:

```yaml
name: flow-state-transitions
description: Verifies node transitions work correctly

# Asserts the entire journey from start to finish
expect_flow_nodes: [greeting, farewell]

phases:
  - send: "Hello! What is 2 + 2?"
    expect_reply_contains:
      - "4"
    # You can also assert the specific node at the end of this turn
    expect_flow_state: greeting

  - send: "Thanks, goodbye!"
    expect_tools:
      - say_farewell
    expect_flow_state: farewell
```

---

## Reliability Testing

LLMs are non-deterministic. The reliability mode lets you assert that a scenario passes *at least M out of N runs*, catching flaky behaviour before it reaches production.

Add a `reliability` block to your YAML:

```yaml
name: reliable-arithmetic
description: Bot consistently answers basic arithmetic
reliability:
  runs: 5       # total number of runs
  min_pass: 4   # minimum required passes

phases:
  - send: "What is 7 plus 3?"
    expect_reply_contains:
      - "10"
```

Reliability mode is **opt-in** — this block is ignored by default and only activated when you pass `--catnip-reliability` (or set `catnip_reliability = true` in `pytest.ini`/`pyproject.toml`). This keeps your regular CI fast while allowing dedicated reliability runs.

```sh
# Run reliability scenarios
pytest tests/cases/ --catnip-reliability

# Combine with other flags
pytest tests/cases/ --catnip-reliability --catnip-show-conversation
```

---

## CLI Options & ini Settings

All options are available both as CLI flags and as `pytest.ini` / `pyproject.toml` settings.

| CLI flag | ini key | Default | Description |
|---|---|---|---|
| `--catnip-show-conversation` | `catnip_show_conversation` | `false` | Print the full conversation log for each test after it runs. Useful for debugging failures. |
| `--catnip-reliability` | `catnip_reliability` | `false` | Enable reliability mode — run scenarios with a `reliability` block N times and assert M passes. |
| `--catnip-overall-timeout` | `catnip_overall_timeout` | `120` | Maximum wall-clock seconds for a single scenario (all phases). Prevents hung pipelines from blocking CI. |
| *(n/a)* | `catnip_cases_directory` | `tests/cases` | Directory scanned for `test_*.yaml` files. |

**Example `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
catnip_cases_directory = "tests/cases"
catnip_show_conversation = true
catnip_overall_timeout = "60"
```

---

## Debugging Failures

Enable conversation logging to see exactly what was exchanged during a failing run:

```sh
pytest tests/cases/ --catnip-show-conversation -x
```

Output will include a conversation block for each test:

```
-------------------------------- conversation ---------------------------------
  USER  Hello! What is 2 + 2?
   BOT  The answer is 4.
  TOOL  check_something
   BOT  Based on that, here's what I found.
```

For reliability runs, all attempts are shown with their pass/fail status so you can inspect which runs succeeded.

---

## Contributing

```sh
# Install dev dependencies
poetry install --with dev

# Run tests
just test

# Lint
just lint

# Format
just format
```

Tests use pytest-catnip to test itself — the `tests/` directory is a live example of a complete setup, including both a plain-pipeline conftest and a pipecat-flows conftest scoped to a subdirectory.

## Running the project

This project uses [`just`](https://github.com/casey/just) recipes to do all the basic operations (testing the package, formatting the code, etc.).

Installation: 

```sh
brew install just
# or
snap install --edge --classic just
```

It requires [poetry](https://python-poetry.org/docs/#installation).

These are the available commands for the justfile:

```
Available recipes:
    help                        # Shows list of recipes.
    venv                        # Generate the virtual environment.
    clean                       # Cleans all artifacts generated while running this project, including the virtualenv.
    test *test-args=''          # Runs the tests with the specified arguments (any path or pytest argument).
    t *test-args=''             # alias for `test`
    test-all                    # Runs all tests including coverage report.
    format                      # Format all code in the project.
    lint                        # Lint all code in the project.
    docs                        # Serve the documentation in a local server.
    pre-commit *precommit-args  # Runs pre-commit with the given arguments (defaults to install).
    spellcheck *codespell-args  # Spellchecks your markdown files.
    lint-commit                 # Lints commit messages according to conventional commit rules.
```

To run the tests of this package, simply run:

```sh
# All tests
just t

# A single test
just t tests/test_dummy.py

# Pass arguments to pytest like this
just t -k test_dummy -vv
```

## Managing requirements

`poetry` is the tool we use for managing requirements in this project. The generated virtual environment is kept within the directory of the project (in a directory named `.venv`), thanks to the option `POETRY_VIRTUALENVS_IN_PROJECT=1`. Refer to the [poetry documentation](https://python-poetry.org/docs/cli/) to see the list of available commands.

As a short summary:

- Add a dependency:

        poetry add foo-bar

- Remove a dependency:

        poetry remove foo-bar

- Update a dependency (within constraints set in `pyproject.toml`):

        poetry update foo-bar

- Update the lockfile with the contents of `pyproject.toml` (for instance, when getting a conflict after a rebase):

        poetry lock

- Check if `pyproject.toml` is in sync with `poetry.lock`:

        poetry lock --check


## Contributing
In this project we enforce [conventional commits](https://www.conventionalcommits.org) guidelines for commit messages. The usage of [commitizen](https://commitizen-tools.github.io/commitizen/) is recommended, but not required. Story numbers (JIRA, etc.) must go in the scope section of the commit message. Example message:

```
feat(JIRA-XXX): add new feature x
```

Merge requests must be approved before they can be merged to the `main` branch, and all the steps in the `ci` pipeline must pass.

This project includes an optional pre-commit configuration. Note that all necessary checks are always executed in the ci pipeline, but
configuring pre-commit to execute some of them can be beneficial to reduce late errors. To do so, simply execute the following just recipe:

```
just pre-commit
```

> [!note] 🔗 Useful links
> - [API Development Guidelines](https://www.notion.so/msdevelopment/Development-Guidelines-623677e75f69473abc743ce1d381eb6b)
>
> - [Code Review Guidelines](https://www.notion.so/msdevelopment/Code-Review-Guidelines-82023d1814b442e486ed38e648c5d86e)
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/scastlara"><img src="https://avatars.githubusercontent.com/u/7606872?v=4?s=50" width="50px;" alt="Sergio Castillo"/><br /><sub><b>Sergio Castillo</b></sub></a><br /><a href="https://github.com/elementsinteractive/pytest-catnip/commits?author=scastlara" title="Code">💻</a> <a href="#ideas-scastlara" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-scastlara" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Rooni"><img src="https://avatars.githubusercontent.com/u/916242?v=4?s=50" width="50px;" alt="Rooni"/><br /><sub><b>Rooni</b></sub></a><br /><a href="https://github.com/elementsinteractive/pytest-catnip/commits?author=Rooni" title="Code">💻</a> <a href="#ideas-Rooni" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!