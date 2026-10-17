# Quick start: from an idea to a private GitHub repo

## One-time setup (2 minutes)

1. Make a GitHub token: GitHub > Settings > Developer settings > Personal access tokens > **Tokens (classic)** >
   Generate new token, tick the **`repo`** box, copy it.
2. In the terminal: `codeagent setup` and paste the token (nothing is shown while you type; it tells you how many
   characters it received). If the paste does not work, use `codeagent setup --visible`, or Ctrl+Shift+V.
   It checks the token, saves it in `~/.codeagent.env` (only you can read it) and tells you which account it found.
3. Check any time with `codeagent whoami`.

## For every new project (5 steps)

1. **Ask an AI to write the plans.** Open any AI chat, paste `docs/PLAN_GUIDE_FOR_AI.md`, then write what you want built
   ("a todo CLI", "an expense tracker library", ...). It answers with plan files, `project.json` and `groups.json`.
2. **Save the files** in one folder, for example `~/projects/expense/` (exact file names as the AI says).
3. **Check the plans:** `codeagent check plan-1-core.json` (once per plan). Fix any `warning:` first.
4. **Build everything:** `codeagent build ~/projects/expense/project.json`
   It creates a **private** repo named after the project, builds all plans, merges the files into a few modules, and
   pushes. Your commits are made as you. You do nothing else.
5. **Open the repo** (the link is printed at the end). Clone it: `git clone https://github.com/<you>/<name>`.
6. **Try the app in your browser:** `cd ~/agent-projects/<name> && python ui.py`, then open http://127.0.0.1:8000
   (see "The small UI" below).

## If it says some steps could not be built

The rest of the project is already built and saved. It wrote `FAILURES.md` and told you the exact commands:

1. Open `<workspace>/.codeagent/FAILURES.md`, copy everything, paste it into any AI chat (or send it to me).
2. The AI answers with a JSON "fix pack". Save it as `fix.json`.
3. `codeagent fix fix.json --workspace <the workspace printed by build>`  (or `codeagent fix -`, paste, Ctrl-D)
4. Run the same `codeagent build ...` command again. It continues with the fixed steps.

## The small UI

`build` also writes `ui.py` into the project: a simple local web page with a form for **every function** of your app.
Pick a function on the left, fill the boxes (they start filled with an example), press **Run**. To chain steps, type a
name in "save result as $" and then use `$name` in any other box (make a record, add it to a list, ask for a summary).
Only your computer can open it. For a project built before this existed:
`codeagent ui ~/agent-projects/<name> --plans plan-1.json plan-2.json --title "My App" --run`.

## Good to know

- Run `build` again any time: it continues and never makes a second repository.
- `--no-github` builds on your computer only. `--dry-run` shows what would happen. `--no-main` pushes only the work branch.
  `--no-ui` skips `ui.py`.
- Secrets are never committed: `.env` is ignored, every commit and the whole history are scanned before a push.
- The workspace is `~/agent-projects/<name>` unless you pass `--workspace`.
