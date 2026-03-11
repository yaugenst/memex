---
name: instruction-improver
description: Audit memex history for recurring instruction gaps, validate them with full-session evidence, and propose high-confidence improvements to CLAUDE.md or AGENTS.md. Use when you want to learn from repeated corrections, failures, or wins before changing standing instructions.
---

# Instruction Improver

Default behavior: run an audit first. Do not edit any instruction file until the
user approves concrete findings.

## What This Skill Is For

Use this skill to:
- Mine `memex` history for recurring failures, corrections, and wins
- Identify instruction gaps behind those incidents
- Propose evidence-backed updates to `CLAUDE.md` and/or `AGENTS.md`

Do not use this skill to:
- Encode one-off annoyances as permanent rules
- Treat profanity or praise as the pattern itself
- Edit instruction files before you have enough evidence

## Targets

This skill may recommend changes to:
- `~/.claude/CLAUDE.md` for cross-project Claude behavior
- A user-level Codex instruction file, if the user's setup has one, for
  cross-project Codex behavior
- A user-level shared instruction file, if the user keeps one, for global rules
  that should apply across tools
- Project `CLAUDE.md` or `.claude/CLAUDE.md` for project-specific Claude behavior
- Project `AGENTS.md` for repo-local Codex or shared engineering instructions

When the right target is unclear, audit first and recommend a target per finding.
If no approved global Codex/shared target exists, mark the finding as a promotion
candidate instead of forcing it into a project-local file.

## Default Mode

Default to **audit-only** unless the user explicitly asks to apply changes.

If the user says "improve my instructions" without naming a file:
1. Ask what to audit:
   - current project
   - all projects
   - a recent time window
   - a specific pain point
2. Produce findings first
3. Ask whether any finding should be applied, and to which file

Do **not** begin by asking which file to edit. First establish whether an edit is
actually justified.

## Workflow

### Phase 1: Collect Candidate Incidents

Start broad, then narrow.

If time window matters, begin with session discovery:

```bash
memex sessions --since <iso|unix> --until <iso|unix> --json-array
```

Then search for candidate failure or success classes. Prefer incident classes over
raw emotion terms.

### Failure Classes

```bash
# Misunderstanding or ignored constraint
memex search "not what I asked|I said|I meant|stop|don't|wrong" --role user --unique-session --limit 30

# Destructive or regressive edits
memex search "you deleted|you removed|you broke|regression|revert|undo" --role user --unique-session --limit 30

# Insufficient investigation
memex search "did you check|inspect|read the file|understand|look at" --role user --unique-session --limit 30

# Poor validation
memex search "did you test|run tests|verify|still broken|doesn't work" --role user --unique-session --limit 30

# Verbosity or output-shape mismatch
memex search "too verbose|shorter|just answer|don't explain|bullet points" --role user --unique-session --limit 30

# Tool or workflow mismatch
memex search "wrong tool|don't use|never use|don't commit|don't push" --role user --unique-session --limit 30
```

### Success Classes

```bash
# Successful completions
memex search "works|working|fixed|solved|done" --role user --unique-session --limit 30

# Praise for a behavior worth reinforcing
memex search "perfect|exactly|nice|great|lgtm|looks good" --role user --unique-session --limit 30

# Workflow approval
memex search "good pr|good summary|thanks for checking|nice catch" --role user --unique-session --limit 20
```

### Broader Retrieval When Lexical Search Is Too Narrow

When exact phrases are too brittle, try semantic or hybrid search with a concrete
incident description:

```bash
memex search "edited before understanding surrounding files" --hybrid --role user --unique-session --limit 20
memex search "failed because it did not verify the fix" --semantic --role user --unique-session --limit 20
```

### Useful Reducers

- `--project <name>`
- `--source claude|codex|opencode`
- `--since <iso|unix>` / `--until <iso|unix>`
- `--top-n-per-session 2`
- `--fields score,ts,session_id,project,snippet`

For project-scoped audits, prefer project filters early. For cross-project audits,
look for patterns that repeat in multiple codebases before recommending a global
rule.

### Phase 2: Validate With Full Context

Do not propose an instruction from search snippets alone.

For promising hits:

```bash
memex session <session_id>
memex show <doc_id>
memex session <session_id> --verbose
memex show <doc_id> --verbose
```

For each candidate incident:
- Read the assistant behavior immediately before the user correction
- Reconstruct the actual failure mode in neutral language
- Separate the surface wording from the root cause

Bad:
- "User said 'wrong'"

Good:
- "Assistant edited before understanding adjacent files"
- "Assistant answered in long-form prose after repeated requests for terse output"
- "Assistant skipped validation and claimed the fix worked"

### Phase 3: Cluster and Score

Group incidents into recurring patterns. Score each pattern on:

- `frequency`
  - `1`: one-off
  - `2`: repeated in independent sessions
  - `3`: recurring pattern
  - `4`: recurring across projects
- `severity`
  - `1`: nuisance
  - `2`: wasted time
  - `3`: incorrect output or regression
  - `4`: unsafe or destructive
- `scope fit`
  - `user-global`
  - `project`
  - `tool-specific`
  - `none`
- `evidence spread`
  - `one-project`
  - `multi-project`
  - `one-tool`
  - `multi-tool`
- `confidence`
  - `low`
  - `medium`
  - `high`
- `promotion state`
  - `watchlist`
  - `local-candidate`
  - `global-candidate`
  - `immediate-rule`

Only propose permanent instructions when:
- At least 2 independent sessions show the same failure mode, or
- A single incident is severe enough that a standing rule is obviously warranted

Reject or down-rank:
- Pure mood signals without an actionable root cause
- Low-severity one-offs
- Project-specific incidents proposed as global rules
- Patterns already covered by current instructions

Prefer local by default. Promote to global only when the evidence clearly travels
outside one repository, or when the rule is obviously a user/tool-wide preference.

### Phase 4: Produce an Audit Report

Return a short list of high-signal findings. Prefer 3 to 5 findings max.

Use this format:

```markdown
## Pattern: <short name>

Type: negative|positive
Frequency: <1-4>
Severity: <1-4>
Evidence spread: one-project|multi-project|one-tool|multi-tool
Confidence: low|medium|high
Promotion state: watchlist|local-candidate|global-candidate|immediate-rule
Recommended target: <~/.claude/CLAUDE.md | user-level Codex file | shared global file | project CLAUDE.md | AGENTS.md>

Evidence:
- <session_id> - <one-line summary>
- <session_id> - <one-line summary>

Root cause:
<what actually went wrong or right>

Proposed instruction:
<the exact candidate rule or edit>

Why this belongs there:
<why the target file and scope are correct>
```

If useful, add a `Do not encode` section for patterns you investigated but rejected.

### Phase 5: Edit Only After Approval

After the user approves specific findings:
1. Read the target file(s)
2. Check for duplicates or contradictions
3. Apply the smallest change that captures the pattern
4. Show the resulting diff or inserted text

If multiple files are involved:
- Keep tool-specific wording separated
- Do not blindly mirror text between `CLAUDE.md` and `AGENTS.md`
- Mirror only when the rule is genuinely shared and project-specific

## Scope Matrix

Use this guidance when recommending where a finding belongs:

- `~/.claude/CLAUDE.md`
  - Cross-project Claude behavior
  - Global communication defaults
  - User-wide preferences that are not repo-specific

- User-level Codex instruction file
  - Cross-project Codex behavior
  - Tool habits that should follow the user across repositories
  - Use this only if the user actually has a global Codex target they maintain

- User-level shared instruction file
  - Cross-tool rules that should apply everywhere
  - Use this only if the user actually keeps a shared global instruction file

- Project `CLAUDE.md` or `.claude/CLAUDE.md`
  - Project-specific Claude workflow
  - Stack or codebase rules specific to this repository

- Project `AGENTS.md`
  - Repo-local engineering rules
  - Codex workflow and tool usage
  - Branch, testing, editing, or file-structure conventions

When uncertain:
- Behavioral and communication patterns lean global
- Technical and repo-specific patterns lean project-level
- Codex- or repo-operation-specific rules usually belong in `AGENTS.md`
- If a rule looks global but no global target exists yet, keep it as a
  `global-candidate` instead of stuffing it into a local file

## Promotion Model

Use these states explicitly:

- `watchlist`
  - interesting, but not enough evidence yet
- `local-candidate`
  - repeated within one project
  - useful for refining the workflow in that repo
- `global-candidate`
  - repeated across projects, or clearly a user/tool-wide preference
  - useful, but waiting for confirmation or an explicit global target
- `immediate-rule`
  - severe enough to encode now, even from a single incident

Promotion heuristic:
- Start local unless evidence clearly says global
- Promote from local to global when the same pattern recurs in multiple projects
- Promote immediately when the rule is clearly about tool behavior or user
  communication preferences rather than repo structure

## Anti-Patterns

Do not:
- Ask "which file should I edit?" before you know whether any edit is warranted
- Treat profanity as the insight instead of a lead
- Promote a single low-severity complaint into a permanent rule
- Promote a one-project workflow scar into a global rule
- Write rules that merely restate symptoms
- Ignore the current instruction files when proposing edits
- Edit by default

## Recommended Flow

1. Clarify the audit scope if the user has not already provided it
2. Gather 10 to 30 candidate incidents
3. Validate the best candidates with full transcripts
4. Produce 3 to 5 high-signal findings
5. Ask whether to apply any of them, and where
