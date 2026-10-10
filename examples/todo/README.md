# Todo list: a small app built with the agent

The agent builds the **pure core** (7 small functions, plan: `tests/examples/todo_core.json`).
The glue that touches the outside world (saving to a file, the command line) is written by hand, here.

```bash
codeagent check tests/examples/todo_core.json          # no warnings expected
codeagent run tests/examples/todo_core.json --workspace ~/agent-projects/todo --live

# when the plan is DONE, add the hand-written glue and its tests:
W=~/agent-projects/todo
cp examples/todo/store.py examples/todo/cli.py examples/todo/__main__.py $W/todo/
cp examples/todo/test_cli.py $W/tests/
cd $W && python -m pytest -q                            # core tests + glue tests

python -m todo add "buy milk"
python -m todo add "call mum"
python -m todo done 2
python -m todo list
python -m todo list --status open
python -m todo rm 1
```

Tasks are saved in `tasks.json` in the folder you run it from (`--file other.json` to change).
If a step fails, read `$W/.codeagent/transcript.md`, fix the plan wording, and rerun: finished steps are skipped.
