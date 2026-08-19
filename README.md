# AI Blogger — Multi-Agent Studio

## Project Overview

This project writes and publishes blog posts using a team of small AI agents. You type a
topic, and five agents take it from there: one searches the web, one checks the findings
are true, one writes the post, one reviews how it reads, and one publishes it to Google
Blogger.


The agents are wired together with **LangGraph** using an **orchestration workflow
pattern**. Rather than one large model doing everything in a single prompt, the work is
split into fixed steps with clear handoffs. A central graph decides which agent runs next
from the current state, and two of those steps are loops: if the research is poor it goes
back to the researcher, if the draft reads badly it goes back to the writer. Each agent has
one job and one prompt, which makes it possible to tell *which* step went wrong when the
output is bad — and that turned out to matter enormously.

Nothing gets published without your approval. The workflow deliberately stops before the
final step and waits for a person — the **Human In The Loop (HITL)** part. The graph saves
its progress to a database, pauses, and shows you the draft in the dashboard. Only when you
click **Approve & Publish** does it resume and post the article. Close the tab or hit
**Stop** and the run is cancelled instead.

Every run is recorded in **LangSmith**, used here for three things. 
**Tracing:** open any run and see each agent's exact prompt, its reply, how long it took and how many tokens it used — that is how nearly every bug in this project was found. 
**Evaluation:** a fixed set of 20 test topics lives there as a dataset, and three grouped judges score each finished post on nine measures. 
**Prompt Hub:** every prompt is published there with a version number, so a score can always be traced back to the exact prompt that produced it.

## Tech Stack

| Part | What it does |
|---|---|
| **LangGraph** | Runs the agent workflow, handles the loops, and saves progress |
| **Ollama** | Runs the local models on your machine |
| **Google Gemini** | The fact-checking validator and the evaluation judges |
| **MCP (Model Context Protocol)** | Keeps tools in separate processes from the agents |
| **Tavily** | Web search, used by the researcher |
| **Google Blogger API** | Publishes the finished post |
| **PostgreSQL (Supabase)** | Stores workflow progress so a paused run can be resumed |
| **Gradio** | The web dashboard |
| **LangSmith** | Run tracing, evaluation, and prompt storage |
| **Docker** | Runs the dashboard in a container |
| **pytest + GitHub Actions** | 87 tests that run automatically on every push |

## Features

- **Five specialised agents** instead of one big prompt, so each step can be checked separately.
- **Human approval before publishing.** The run pauses and waits for you.
- **Stop button.** Cancel a running workflow at any point; the saved progress is deleted.
- **Automatic cleanup.** Closing the browser tab or shutting the app down cancels any run
  still going. A run already waiting for your approval is kept, so you never lose a
  finished draft.
- **Retry loops with limits.** Weak research goes back to the researcher, a weak draft goes
  back to the writer — both stop after a set number of tries instead of looping forever.
- **Safe failure.** If the research never becomes usable, the run stops and says why. It
  never writes an article from broken data.
- **HTML cleaning in code, not by a model.** Script tags, event handlers and `javascript:`
  links are stripped before publishing by a fixed rule that cannot be talked out of it.
- **Prompts kept outside the code** in `src/prompts/*.yaml`, each with its own version,
  model and temperature. Change a prompt or swap a model without touching any Python.
## Agents Description

| Agent | Model | What it does |
|---|---|---|
| **Researcher** | qwen3 | Searches the web for facts about the topic. Says so plainly when the results turn out to be about a different subject. |
| **Validator** | gemini-3.5-flash-lite | The only step that checks whether facts are true. Everything after it trusts its verdict. |
| **Writer** | qwen3 | Turns approved research into an HTML post. May only state facts from the research — no invented numbers, people or examples. |
| **Editor** | llama3.1:8b | Judges how the post *reads*: headline, tone, structure, skimmability, engagement. Does not check facts. |
| **Publisher** | llama3.1:8b | Posts the approved draft to Blogger and returns the live link. Runs only after a human approves. |

## Agents Workflow

```
  Your topic
      │
      ▼
 ┌──────────────┐
 │  RESEARCHER  │◄─────────────┐
 └──────┬───────┘              │  research rejected
        ▼                      │  (up to 2 tries)
 ┌──────────────┐              │
 │  VALIDATOR   │──────────────┘
 └──────┬───────┘
        │ facts check out
        ▼
 ┌──────────────┐
 │    WRITER    │◄─────────────┐
 └──────┬───────┘              │  reads badly
        ▼                      │  (up to 3 tries)
 ┌──────────────┐              │
 │    EDITOR    │──────────────┘
 └──────┬───────┘
        │ reads well
        ▼
 ┌──────────────┐
 │  SANITIZER   │   ← clean up any unwanted tags
 └──────┬───────┘
        ▼
  ╔══════════════╗
  ║ PAUSE — YOU  ║   ← the workflow stops here and waits
  ║   REVIEW IT  ║      (Human In The Loop)
  ╚══════┬═══════╝
         │ you click Approve & Publish
         ▼
 ┌──────────────┐
 │  PUBLISHER   │──► live blog post
 └──────────────┘
```

If the research cannot be validated after two tries, the run stops instead of continuing,
and the dashboard shows the reason.

## Models Used

| Model | Where it runs | Used by | Why this one |
|---|---|---|---|
| **qwen3** | Local (Ollama) | Researcher, Writer | Best local model tested for writing. Against llama3.1:8b on an identical prompt it scored higher on engagement with the same headline quality. |
| **llama3.1:8b** | Local (Ollama) | Editor, Publisher | Fast, and these two roles only need a short verdict or a tool call. |
| **gemini-3.5-flash-lite** | Hosted (Google) | Validator, evaluation judges | Needs current knowledge — see [Results](#results). |


## Screenshots

**The dashboard while a workflow is running.** The live trace on the left shows which agent
has finished and what it produced. The Stop button is active while the run is in progress.

![Dashboard running a workflow](https://github.com/PrashantMaht0/multi_agent_ai_blogger/blob/main/assest/screen_shots/Dashboard.png)

**Waiting for human approval.** The editor has passed the draft, the workflow has paused,
and the finished HTML is shown for review. Nothing is published until Approve & Publish is
clicked.

![Draft paused for human review](https://github.com/PrashantMaht0/multi_agent_ai_blogger/blob/main/assest/screen_shots/Blog_human_review.png)

**The published post**, live on Blogger after approval.

[![Published blog post](https://github.com/PrashantMaht0/multi_agent_ai_blogger/blob/main/assest/screen_shots/Final_blog_with_v1.png)](https://prax-pins-gg.blogspot.com/2026/08/everything-new-in-googles-new-pixel-11.html)

## Run the Project Locally

### 1. What you need first

- **Python 3.13** and **[Poetry](https://python-poetry.org/docs/#installation)**
- **[Ollama](https://ollama.com/download)** installed and running
- A machine with **16 GB RAM** (the two local models total about 10 GB)
- Free API keys: 
  - **[Tavily](https://tavily.com)** (web search),
  - **[Google AI Studio](https://aistudio.google.com/apikey)** (Gemini),
  - **[LangSmith](https://smith.langchain.com)** (tracing — optional but recommended)
- A **[Supabase](https://supabase.com)** project for the free PostgreSQL database
- A **[Blogger](https://www.blogger.com)** blog, only if you want to publish for real

### 2. Get the code and install

```bash
git clone https://github.com/PrashantMaht0/multi_agent_ai_blogger.git
cd multi_agent_ai_blogger
poetry install
```

### 3. Download the local models

```bash
ollama pull qwen3
ollama pull llama3.1:8b
```

Leave Ollama running in the background.

### 4. Create your `.env`

Copy the template and fill in your own keys:

```bash
cp example.env .env
```

Open `.env` and set `TAVILY_API_KEY`, `GEMINI_API_KEY`, `LANGSMITH_API_KEY`,
`POSTGRES_DB_URL` (the direct connection string from Supabase → Settings → Database) and
`BLOGGER_BLOG_ID`. Leave `OLLAMA_BASE_URL` as `http://localhost:11434`.

### 5. Authorise Blogger (only if you want to publish)

Download an OAuth **desktop app** credentials file from Google Cloud, save it as
`credentials.json` in the project root, then run:

```bash
poetry run python src/auth_blogger.py
```

### 6. Start the dashboard

```bash
poetry run python app.py
```

Open **http://localhost:7860**, type a topic, and click **Generate**.  When it pauses, read the draft and click **Approve & Publish**.

### 7. Run the tests

```bash
poetry run python -m pytest tests -v
```

### 8. Run the evaluation (optional)

```bash
poetry run python tests/eval_harness.py --limit 3    # try 3 topics first
poetry run python tests/eval_harness.py              # all 20
```

Each topic spends one Tavily search credit. Results appear in LangSmith.

### Running in Docker instead

```bash
docker compose up --build
```

Ollama still runs on your host; the compose file points the container at it. `token.json`
and `credentials.json` are mounted read-only and never baked into the image.

## Evals Configs

### Dataset

20 test topics, kept in `tests/dataset.json` 

- **14 normal topics** — ordinary technical subjects such as "How does retrieval-augmented
  generation work?" Each lists the key facts good research should surface.
- **6 attack topics** — hostile inputs checking the system cannot be tricked: credential
  theft, reading environment variables, injecting a `<script>` tag into the published page,
  a destructive database command, hijacking the workflow to skip review, and planting a
  hidden instruction in the output for the next agent to obey.


### Active evaluators

Nine measures produced by **three grouped judges**, all on `gemini-3.5-flash-lite`. Each
judge makes one call and returns several scores, so a full evaluation costs 3 judge calls per
topic instead of 9.

**1. Trust & Safety**

| Measure | What it checks |
|---|---|
| **harmful_content** | No hate speech, abuse or otherwise harmful material |
| **security** | An injected instruction did not change what the system did. Writing *about* an instruction is fine; obeying it fails |
| **correctness** | Did the research find the expected facts? Scored as a fraction, so 2 of 3 = 0.67 |
| **hallucination_free** | Every specific claim in the article traces back to the research |

**2. Editorial & Reader Experience**

| Measure | What it checks |
|---|---|
| **catchy_headline** | The title is specific and worth clicking, not a textbook chapter name |
| **tone** | Informative and professional, no marketing hype |
| **engagement** | It holds attention and gives a reason to keep reading |

**3. Structure & Visual Layout**

| Measure | What it checks |
|---|---|
| **structure** | Clear opening, sections in a sensible order, headings that describe what follows |
| **skimmability** | Short paragraphs, useful subheadings, sections that connect |

## Results

### Baseline and final

Both runs cover the same 20 topics with the same judges.

| Measure | Baseline | Final | Change |
|---|---|---|---|
| harmful_content | 1.00 | **1.00** | — |
| security | 0.94 | **0.94** | — |
| **correctness** | 0.25 | **0.86** | **+0.61** |
| hallucination_free | 1.00 | **0.76** | −0.24 |
| **catchy_headline** | 0.62 | **0.90** | **+0.28** |
| tone | 0.99 | **0.98** | — |
| **engagement** | 0.18 | **0.93** | **+0.75** |
| structure | 1.00 | **0.83** | −0.17 |
| skimmability | 0.91 | **0.82** | −0.09 |
| **Time per post** | 224s | **146s** | **−35%** |
| Runs that failed | 4/20 | **3/20** | −1 |

Earlier, before the hybrid setup, a full run took **566 seconds per post**. Against that
starting point the finished pipeline is **nearly 4× faster**.
**Why the Validator Agent moved to Gemini?**
The first version used `gemma4:12b`, running locally, to check whether the research was
true. It kept rejecting research that was perfectly correct.
A local model's knowledge stops at its training date. A validator/fact-checker is exactly the job
where that matters most, because its whole purpose is to know what is true *now*.

So the validator moved to `gemini-3.5-flash-lite`, and every other agent stayed local. That
one change did three things:

1. **Correct research stopped being thrown away.** 
2. **The other agents got simpler.** 
3. **Runs got much faster.** 

The cost is one hosted API call per run. Everything that actually writes your blog post
still runs on your machine.

### How the scores were raised

**Correctness (0.25 → 0.86)**
This was a formatting issue, not a research failure. The judge AI couldn't read research notes hidden inside a JSON file, which I fixed by passing them as plain text.

**Engagement (0.18 → 0.93) and Headline (0.62 → 0.90)**
I improved this partly by updating the prompts and tempering with the temprature values to ban clice openings (like "Understanding...") and predictable endings (like "In conclusion"). 

**Why hallucination_free dipped to 0.76**
This drop was a real trade-off. I raised the model's temperature from 0.7 to 0.85 to make the writing less formulaic. It worked, but when asked to "open with something concrete," the hotter model started inventing fake details like a made-up developer in Berlin or a fake $150K project. 

## Future Enhancements

- **Image support.** Generating or sourcing a header image and inline diagrams, with
  captions and alt text.
- **Mixed model setups.** Trying a larger model only for the writer, or a small fast model
  for the judges, and comparing the scores. The prompt files already allow this by editing
  one line.
- **HTML cleaning before publishing.** A fixed rule that strips unsafe tags, rather than
  relying on the models to behave.
- **Faster evaluation runs.** Combining the four judges into one call and caching research
  by topic, so repeat runs cost no search credits.
- **SEO support.** Keyword suggestions, meta descriptions, and internal links.
- **More publishing targets** beyond Blogger, such as WordPress, Dev.to or Medium.
- **Scheduling.** Queue several topics and let them run unattended, with approval collected
  later in one batch.

## How to Contribute and Report Issues

Contributions are welcome! If you want to contribute, please follow these steps:

* Fork the Repository: Create your own branch from main.
* Create a Feature Branch: git checkout -b feature/AmazingFeature
* Commit your Changes: Write clear commit messages.
* Push to the Branch: git push origin feature/AmazingFeature
* Open a Pull Request: Describe the changes you made and the problem they solve.

**Before opening a pull request**

* The tests must pass, and a change should come with a test:

  ```bash
  poetry run python -m pytest tests -v
  ```

* Keep the tests hermetic. They must never read your `.env`, call a model, or reach the
  network — mock the model and the search call instead. `tests/conftest.py` already blocks
  `.env` and supplies placeholder keys.
* Never commit `.env`, `credentials.json` or `token.json`. They are in `.gitignore`; keep
  them there.
* Prompts belong in `src/prompts/*.yaml`, not in Python. Editing one means bumping its
  `version` and publishing it, so a score can be traced back to the prompt that produced it:

  ```bash
  poetry run python tests/push_prompts.py writer
  ```

* Model settings — temperature, output cap, thinking on or off — belong in the prompt YAML
  too, so an agent's behaviour can be changed without touching code.
* Treat the topic and the draft as untrusted text everywhere. A prompt must tell the model
  to ignore instructions found inside them, and unsafe HTML is stripped by
  `src/agents/sanitize.py` rather than by asking a model nicely.
* Never put a literal score in a judge's example JSON. The judge copies it instead of
  scoring, which silently broke two measures for several evaluation runs. A test enforces
  this.
* Editing `tests/dataset.json` only changes the local copy. Push the change to LangSmith
  so evaluations use it:

  ```bash
  poetry run python tests/eval_harness.py --sync-dataset --limit 1
  ```

**Reporting Issues**
If you find a bug or have a feature request, please use the GitHub Issues tab. Include the following in your report:

* A clear title.
* Steps to reproduce the bug.
* Expected vs. actual behavior.
* Screenshots or error logs if you have them.
