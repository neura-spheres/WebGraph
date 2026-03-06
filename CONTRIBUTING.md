# Contributing to NeuraSearch

First off, thank you for taking the time to read this. It genuinely means a lot that you are considering contributing to this project.

NeuraSearch is a student-built, open-source search engine backend. The codebase is intentionally kept readable and well-documented so that anyone, regardless of experience level, can understand what each part does and contribute meaningfully. Whether you are here to fix a typo, squash a bug, or implement something from the roadmap, all of it is useful.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Submitting a Pull Request](#submitting-a-pull-request)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [What We Are Looking For](#what-we-are-looking-for)
- [What to Avoid](#what-to-avoid)
- [Getting Help](#getting-help)

---

## Code of Conduct

This project follows a simple rule: be respectful. Everyone here is learning something and doing their best. Criticism of code is fine and encouraged. Criticism of people is not.

If you see behavior that makes this a hostile or unwelcoming space, please open an issue and flag it.

---

## How to Contribute

### Reporting Bugs

If you found something broken, please open a GitHub issue and include:

- A clear title describing the problem
- Steps to reproduce it (be as specific as possible)
- What you expected to happen
- What actually happened
- Your Python version and operating system
- Any relevant error messages or stack traces

The more detail you include, the faster it can be fixed.

### Suggesting Features

If you have an idea for something new, open an issue with the `enhancement` label and describe:

- What problem it solves
- How you imagine it working at a high level
- Whether you are willing to work on it yourself

We do not promise to implement every suggestion, but all ideas are read and considered.

### Submitting a Pull Request

1. **Fork the repository** and create a new branch from `main`.

   ```bash
   git checkout -b your-feature-name
   ```

2. **Make your changes.** Try to keep each pull request focused on one thing. If you find yourself fixing five unrelated things at once, consider splitting them into separate PRs.

3. **Test your changes manually.** Run a short crawl, index, and search to make sure nothing is broken.

   ```bash
   python main.py crawl https://en.wikipedia.org/wiki/Python --limit 50 --depth 2
   python main.py index
   python main.py pagerank
   python main.py serve
   # Then try a search at http://localhost:8000/search?q=python
   ```

4. **Write a clear PR description** that explains:
   - What you changed
   - Why you changed it
   - How to test or verify the change

5. **Open the pull request** against the `main` branch.

6. **Respond to feedback.** Code review may take a few days. If changes are requested, update your branch and push again. We are not trying to be harsh, just trying to keep the code clean and consistent.

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/neura-spheres/NeuraSearch.git
cd NeuraSearch/backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python main.py setup
```

There are no automated tests yet (this is on the roadmap). For now, manual smoke testing is the way to verify things work.

---

## Code Style

We do not enforce a strict formatter at the moment, but please follow these conventions when writing code:

- Use 4-space indentation throughout.
- Keep line length under 100 characters where reasonable.
- Name variables and functions clearly. Avoid single-letter names except in obvious contexts like loop counters.
- Add a docstring to any new class or function that is not immediately obvious.
- Group imports into three sections separated by blank lines: standard library, third-party packages, local modules.
- Do not leave commented-out code in your pull request. If you want to preserve something for reference, add a note in the PR description instead.

The existing code in this repo is the best style guide. When in doubt, look at how similar things are already written and match that pattern.

---

## Commit Messages

Please write commit messages that are short and descriptive. A good commit message finishes the sentence "This commit will...".

Good examples:
```
Add anchor text diversity signal to Penguin scorer
Fix crawler not respecting nofollow on relative links
Update BM25 refresh to run after PageRank pipeline completes
```

Not so good:
```
fix stuff
update
wip
```

If your change is large or non-obvious, add a longer description after the short summary, separated by a blank line.

---

## What We Are Looking For

These are areas where contributions would be especially valuable:

- **Bug fixes.** If something is broken or behaves unexpectedly, fixing it is always a welcome contribution.
- **Documentation improvements.** If something in the README or inline docstrings was unclear or wrong, please correct it.
- **Roadmap items.** The [Roadmap section of the README](README.md#roadmap) lists specific features we want to add. Any of those are fair game.
- **Performance improvements.** If you can make the crawl faster, the index smaller, or the search query cheaper without breaking correctness, that is great.
- **Better test coverage.** We currently have no automated tests. Adding unit tests for the text processor, BM25 computation, Panda scorer, or Penguin scorer would be a huge improvement.

---

## What to Avoid

To keep things from getting messy, please avoid the following in your pull requests:

- Large refactors that touch many files without a clear motivation. If you think a big refactor is needed, open an issue to discuss it first.
- Adding new dependencies without a good reason. The project currently runs on a small set of well-known libraries and we would like to keep it that way.
- Changes that break the CLI interface or the API response format without a discussion first. Other people may have integrations that depend on the current behavior.
- Committing generated files, SQLite databases, or the contents of the `data/` directory.

---

## Getting Help

If you are stuck on something or not sure whether your idea is a good fit, open an issue and ask. There is no such thing as a bad question here.

You can also reach the maintainers by tagging them in any issue or pull request.

---

Thank you again for contributing. Every improvement, no matter how small, makes this a better project for everyone who learns from it.
