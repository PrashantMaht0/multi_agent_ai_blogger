"""
tests/push_prompts.py
Publishes src/prompts/*.yaml to the LangSmith Prompt Hub, stamped with the git commit
the prompt text came from.

    python tests/push_prompts.py            # push all five
    python tests/push_prompts.py writer     # push one

The YAML files stay the source of truth; the hub is the versioned record you can diff
between eval sweeps. Not collected by pytest: the filename does not match test_*.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

load_dotenv()

from src.prompts import PROMPTS_DIR, load_prompt

AGENTS = ["researcher", "validator", "writer", "editor", "publisher"]


def git_commit() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", str(PROMPTS_DIR)], text=True).strip()
    return f"{sha}-dirty" if dirty else sha


def push(client: Client, name: str, commit: str) -> str:
    prompt = load_prompt(name)

    # mustache, so the literal JSON braces inside these prompts are not treated as variables
    template = ChatPromptTemplate.from_template(prompt.template, template_format="mustache")

    url = client.push_prompt(
        f"ai-blogger-{name}",
        object=template,
        description=f"{prompt.description.strip()} (model {prompt.model}, temperature {prompt.temperature})",
        tags=[f"v{prompt.version}"],
        commit_description=f"src/prompts/{name}.yaml v{prompt.version} @ git {commit}",
    )
    print(f"{name:11} v{prompt.version}  ->  {url}")
    return url


def main():
    parser = argparse.ArgumentParser(description="Publish prompts to the LangSmith Prompt Hub.")
    parser.add_argument("names", nargs="*", choices=AGENTS,
                        help="Which prompts to push. Defaults to all.")
    args = parser.parse_args()

    commit = git_commit()
    if commit.endswith("-dirty"):
        print(f"Warning: src/prompts has uncommitted changes; tagging as {commit}")

    client = Client()
    print(f"Pushing prompts at git {commit}\n")
    for name in (args.names or AGENTS):
        push(client, name, commit)


if __name__ == "__main__":
    main()
