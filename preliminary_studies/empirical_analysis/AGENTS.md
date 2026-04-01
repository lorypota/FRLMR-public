# Empirical Analysis Guide

## First section - DO NOT EDIT THIS

This file is in part a duplication of AGENTS.md in root because some agents might be opened directly in this folder and need this specific information without the need to read the previous file.

### IMPORTANT NOTES - DO NOT EDIT THIS

Ask for clarification whenever you're unsure, your goal is to communicate and iterate, there will never be a perfect response, but it's important to keep improving it as much as possible. Try to be straightforward, concise but not overly. Avoid using an excessive amount of adjectives and use simple words instead of overly complex ones unless necessary. Never use dashes (—) for parenthetical statements and use ":" only if necessary, not to add emphasis. Try not to be sensationalistic but instead try to generally be humble.

I want targeted and concise changes, always look for trimmings to simplify things. Feel free to edit other sections of this document if necessary, but always tell me in chat before doing so. NEVER EDIT THIS FIRST SECTION. Before making or asking for changes, ask yourself: Why would this edit be useful to an agent working on the code?

### WRITING TIPS - DO NOT EDIT THIS

When the user makes an error or misunderstands something:

- Correct directly and clearly without being overly supportive
- Don't use phrases like "Good catch!", "Great question!", "You're right to question this"
- Simply state "No, that's incorrect" and explain the correct interpretation

Do NOT use these words unless extremely necessary and try to use simpler terms without adjectives: delve, delves, delved, delving, intricate, intricacies, underscore, underscores, underscoring, showcasing, showcases, realm, pivotal, crucial, comprehensive, meticulous, meticulously, groundbreaking, advancements, aligns, boasts, comprehending, surpassing, surpasses, emphasizing, garnered, noteworthy, notable, commendable, innovative, invaluable, versatile, potent, ingenious, landscape, unparalleled, multifaceted, nuanced, robust, streamline, transformative, leverage, harness, utilize, navigate, foster, enhance, facilitate, Furthermore, Moreover, Additionally, Notably, Importantly, tapestry, interplay, paradigm, cornerstone, holistic, synergy, ever-evolving.

For Writing Structure:

1. State your main point first - what's your argument or finding?
2. Put your thesis in the opening paragraph, not after background information
3. Skip the "throughout history" or "in recent years" openings

For Supporting Arguments: 4. Signal your evidence clearly with phrases like "This is supported by..." or "The evidence shows..." 5. Preview your main evidence points in your introduction 6. Your supporting points should build on each other, not stand alone as separate justifications -> VERY IMPORTANT! 7. Avoid listing multiple independent lines of evidence that don't connect -> VERY IMPORTANT! (stems from 6.)

For Evidence and Citations: 8. Each claim needs backing. When you cite something, explain why it matters to your argument 9. Define specialized terms when you introduce them. If there's debate about a definition, state which one you're using and why

For Counterarguments: 10. Address real challenges to your argument, not weak strawman versions 11. Counterarguments should target your evidence or reasoning, not just restate an opposing view

Writing examples:
Bad Opening:
"Research into climate patterns has been conducted for decades. Scientists have used various methods to study temperature changes. This paper will explore some of these findings..."
Good Openings:
"Global temperatures have risen 1.1°C since pre-industrial times, primarily due to human CO2 emissions. This conclusion rests on three converging lines of evidence: ice core data, satellite measurements, and atmospheric modeling."
"The peer review system improves research quality less than commonly assumed. While it catches obvious errors, studies show it fails to detect more serious methodological flaws, and may actually slow scientific progress."

Bad Counterargument:
"Some might disagree with this interpretation, but the data clearly supports my view."
Good Counterargument:
"However, Smith (2022) argues that satellite measurements may contain systematic errors due to orbital decay. If true, this would reduce the confidence interval for temperature estimates by approximately 0.2°C."

Good Response:
"Yet orbital decay corrections have been standard practice since 2015 (Jones et al., 2015), and are already incorporated into the datasets cited."

### Use UV - DO NOT EDIT THIS

Use `uv` for anything related to this project.

- Create or sync the environment with `uv sync` (only run this if necessary).
- Run scripts with `uv run ...`.
- Add or update dependencies with `uv add ...` or `uv remove ...`.
- Run tooling with `uv run ruff check .` and `uv run ruff format .`.
- Do not introduce ad hoc `pip install`, bare `python script.py`.

## Scope Of This Folder

Read `README.md` first before making changes in this folder. It is the source of truth for folder structure, scripts, workflow, outputs, and run commands. Do not repeat README content here.

This file should only hold agent-specific guidance that helps with editing or reviewing code.

## Agent-Specific Rules

- Prefer keeping current script entry points and output layout stable.
- If you change output names, paths, or generated-page behavior, check `README.md` and update it if needed.
- For `map_den_haag.py`, decide first whether a change belongs in the top-level script or under `internal/area_visualizations/`.
- Try to understand fully requests by user and ask clarifying questions and aim for small targeted changes ONLY if you understand the final goal of user and if what you're doing would specifically answer the user question concretely.
