# AGENTS.md (Local, Personal Workflow)

Local instructions for Codex sessions in this clone.

## AGENTS File Policy
- `AGENTS.md` is tracked on `yaugenst-flex/personal-main` and should be present in personal worktrees.
- Do not include `AGENTS.md` in upstream PR branches.
- If an upstream PR branch contains `AGENTS.md`, drop it before pushing:
  - `git restore --staged AGENTS.md`
  - `git checkout -- AGENTS.md`

## Remotes and Branches
- `upstream`: `nicosuave/memex` (canonical)
- `origin`: `yaugenst/memex` (fork)
- `main` tracks `upstream/main`
- `yaugenst-flex/personal-main` is the default local integration branch and tracks `origin/yaugenst-flex/personal-main`

## Default Working Mode
- In this worktree, default to `yaugenst-flex/personal-main` for personal/local development.
- Merge local improvements into `yaugenst-flex/personal-main`.
- Keep commits small and topic-focused.

## Upstream Contribution Flow
When a fix should go upstream:
1. Start clean from `upstream/main` on a new branch: `yaugenst-flex/fix-<topic>`.
2. Cherry-pick the relevant commit(s) from `yaugenst-flex/personal-main` if needed.
3. Run validation (`build`, `test`, `clippy`, `fmt`).
4. Push to fork and open PR to `nicosuave/memex:main`.

## Sync Routine
```bash
git fetch --all --prune
git switch yaugenst-flex/personal-main
git rebase upstream/main
git push
```

## Worktree Rule
- A branch can only be checked out in one worktree at a time.
- Keep `yaugenst-flex/personal-main` checked out in only one primary worktree.

## Local Env Conventions
- Use untracked `.memex-dev.env` for per-worktree defaults.
- Source it before memex commands: `source ./.memex-dev.env`.
- Keep local data isolated with `MEMEX_DEV_ROOT="$PWD/.memex-dev"`.
- Always pass `--root "$MEMEX_DEV_ROOT"` to `memex` commands.
- Default development model: `MEMEX_MODEL="minilm"`.
- Optional macOS tuning: `MEMEX_COMPUTE_UNITS=ane|gpu|cpu` (no effect on Linux; no effect for `potion`).

## Global Install (stow)
- Global `memex` in `~/.local/bin` is managed via stow from `/Users/yannick/code/memex`.
- Worktree sessions may build local binaries for testing, but should not run stow or update `~/.local/bin/memex` unless explicitly requested.
- Treat `/Users/yannick/code/memex` as the canonical source for global install changes.

## Safety
- Do not run destructive git commands unless explicitly requested.
- Do not commit local-only files (`.memex-dev.env`, `docs/dev-workflow.md`).
