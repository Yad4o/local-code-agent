"""Command line: template | check | doctor | smoke | bench | run | daemon"""
import argparse
import ast
import logging
import logging.handlers
import os
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

from codeagent import __version__
from codeagent.bench import PROBLEMS, run_bench
from codeagent.config import Config, ConfigError, load_config
from codeagent.daemon import Daemon
from codeagent.llm import LLMError, list_models, make_client
from codeagent.parsing import parse_file_blocks
from codeagent.plan import PlanError, load_plan, template_text
from codeagent.prompts import SYSTEM
from codeagent.resources import ResourceGuard
from codeagent.runner import Runner
from codeagent.state import WorkspaceBusy
from codeagent.workspace import Workspace

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_PAUSED, EXIT_BUSY = 0, 1, 2, 3, 4


class _Stopper:
    """First Ctrl+C/SIGTERM = finish the current call and stop cleanly; second = abort now."""

    def __init__(self):
        self.flag = False
        self.count = 0
        self._old: dict = {}

    def __call__(self) -> bool:
        return self.flag

    def _handle(self, signum, frame):
        self.count += 1
        self.flag = True
        if self.count >= 2:
            raise KeyboardInterrupt

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handle)

    def restore(self):
        for sig, old in self._old.items():
            signal.signal(sig, old)
        self._old.clear()


def _make_log(out, log_file: Path | None):
    """Print with a timestamp and also append to a size-capped rotating log file."""
    logger = None
    handler = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3,
                                                       encoding="utf-8")
        logger = logging.getLogger(f"codeagent.{id(handler)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        out(line)
        if logger:
            logger.info(line)

    def close():
        if handler:
            logger.removeHandler(handler)
            handler.close()

    return log, close


def _load(args, out):
    cfg_path = args.config
    if cfg_path is None and Path("codeagent.toml").is_file():
        cfg_path = "codeagent.toml"
    return load_config(cfg_path)


def _check_test_command(cfg: Config, plan) -> None:
    first = shlex.split(plan.test_command)[0]
    if first not in cfg.allowed_commands:
        raise PlanError(f"test_command {first!r} is not in allowed_commands {cfg.allowed_commands}")


# ------------------------------------------------------------------ commands

def cmd_template(args, out, **_):
    if args.out:
        dest = Path(args.out)
        if dest.exists() and not args.force:
            out(f"{dest} already exists (use --force to overwrite)")
            return EXIT_USAGE
        dest.write_text(template_text(), encoding="utf-8")
        out(f"wrote {dest}")
    else:
        out(template_text())
    return EXIT_OK


def cmd_check(args, out, **_):
    cfg = _load(args, out)
    plan = load_plan(args.plan)
    _check_test_command(cfg, plan)
    out(f"Plan OK: {plan.title}  ({len(plan.steps)} step{'s' if len(plan.steps) != 1 else ''})")
    out(f"  test command: {plan.test_command}")
    for i, s in enumerate(plan.steps, 1):
        out(f"  {i}. {s.title}")
        out(f"       tests: {', '.join(s.tests)}")
        out(f"       files: {', '.join(s.files)}")
        out(f"       commit: {s.commit}")
    return EXIT_OK


def cmd_doctor(args, out, **_):
    cfg = _load(args, out)
    failed = False

    def ok(m): out(f"[ok]   {m}")
    def warn(m): out(f"[warn] {m}")
    def fail(m):
        nonlocal failed
        failed = True
        out(f"[FAIL] {m}")

    if sys.version_info >= (3, 11):
        ok(f"python {sys.version.split()[0]}")
    else:
        fail(f"python {sys.version.split()[0]} is too old (need 3.11+)")
    ok("git found") if shutil.which("git") else fail("git is not installed (sudo apt install git)")

    out(f"       backend={cfg.backend} url={cfg.base_url} model={cfg.model} num_ctx={cfg.num_ctx}")
    try:
        models = list_models(cfg)
    except LLMError as e:
        fail(str(e))
    else:
        wanted = {cfg.model, f"{cfg.model}:latest"}
        if wanted & set(models):
            ok(f"model {cfg.model} is installed")
        else:
            avail = ", ".join(models) or "(none)"
            fail(f"model {cfg.model!r} is not installed. Installed: {avail}. "
                 "Set `model = ...` in your config or CODEAGENT_MODEL.")

    guard = ResourceGuard(cfg, Path.cwd())
    mem, bat, temp, disk = guard.free_memory_mb(), guard.battery(), guard.max_temp_c(), guard.free_disk_mb()
    out(f"       memory free: {mem if mem is not None else 'n/a'} MB | "
        f"battery: {f'{bat[0]}% ' + ('(discharging)' if bat[1] else '(plugged in)') if bat else 'none'} | "
        f"cpu temp: {f'{temp:.0f}C' if temp is not None else 'n/a'} | "
        f"disk free: {disk if disk is not None else 'n/a'} MB")
    problems = guard.problems()
    for p in problems:
        warn(p + "  (the agent would pause right now)")
    if not problems:
        ok("resources look fine")
    return EXIT_FAILED if failed else EXIT_OK


def cmd_smoke(args, out, **_):
    """Can this model follow the FILE-block output format at all? (needs the model server)"""
    cfg = _load(args, out)
    client = make_client(cfg)
    prompt = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": "Write the file math_utils.py containing a function "
                                          "add(a, b) that returns the sum of a and b."}]
    good = 0
    for i in range(1, args.tries + 1):
        try:
            reply = client.chat(prompt)
            blocks = {b.path: b.content for b in parse_file_blocks(reply).blocks}
            tree = ast.parse(blocks["math_utils.py"])
            assert any(isinstance(n, ast.FunctionDef) and n.name == "add" for n in ast.walk(tree))
            good += 1
            out(f"  try {i}: ok")
        except LLMError as e:
            out(f"  try {i}: model error: {e}")
        except (KeyError, SyntaxError, AssertionError):
            out(f"  try {i}: reply did not follow the format")
    out(f"format compliance: {good}/{args.tries}")
    if good / args.tries < 0.6:
        out("This model struggles with the output format. Try: temperature = 0 in the config, "
            "a larger model, or simpler/smaller plan steps.")
        return EXIT_FAILED
    return EXIT_OK


def cmd_bench(args, out, *, llm_factory, **_):
    """Raw model test: no agent, no harness. Can the model write these functions at all?"""
    cfg = _load(args, out)
    known = {p.key for p in PROBLEMS}
    unknown = [k for k in (args.only or []) if k not in known]
    if unknown:
        out(f"unknown problem(s): {', '.join(unknown)}. Choose from: {', '.join(sorted(known))}")
        return EXIT_USAGE
    temperature = args.temperature if args.temperature is not None else 0.4
    run_bench(cfg, llm_factory(cfg), tries=args.tries, temperature=temperature, only=args.only, out=out)
    return EXIT_OK


def cmd_run(args, out, *, llm_factory, guard_factory, **_):
    cfg = _load(args, out)
    plan = load_plan(args.plan)
    _check_test_command(cfg, plan)
    ws_path = args.workspace or plan.workspace
    if not ws_path:
        out("no workspace: pass --workspace DIR or add `workspace = \"DIR\"` to the plan")
        return EXIT_USAGE
    ws_path = Path(ws_path).expanduser()
    if args.dry_run:
        out(f"[dry run] would build '{plan.title}' in {ws_path}")
        for i, s in enumerate(plan.steps, 1):
            out(f"  {i}. {s.title}  ->  {s.commit}")
        return EXIT_OK

    ws = Workspace(ws_path, cfg.max_file_bytes)
    log, close = _make_log(out, ws.root / ".codeagent" / "agent.log")
    stopper = _Stopper()
    stopper.install()
    try:
        runner = Runner(cfg, plan, ws, llm_factory(cfg),
                        guard=guard_factory(cfg, ws.root) if guard_factory else None,
                        log=log, should_stop=stopper)
        report = runner.run()
    except WorkspaceBusy as e:
        out(str(e))
        return EXIT_BUSY
    except KeyboardInterrupt:
        out("aborted")
        return 130
    finally:
        stopper.restore()
        close()
    out(report.summary())
    return {"done": EXIT_OK, "failed": EXIT_FAILED, "paused": EXIT_PAUSED}[report.outcome]


def cmd_daemon(args, out, *, llm_factory, guard_factory, **_):
    cfg = _load(args, out)
    inbox = Path(args.inbox).expanduser()
    inbox.mkdir(parents=True, exist_ok=True)
    log, close = _make_log(out, inbox / "agent.log")
    stopper = _Stopper()
    stopper.install()
    try:
        d = Daemon(cfg, inbox, llm_factory=llm_factory,
                   guard_factory=guard_factory or (lambda c, root: ResourceGuard(c, root)),
                   log=log, should_stop=stopper, settle_seconds=0 if args.once else 2.0)
        if args.once:
            d.prepare()
            d.run_once()
        else:
            d.run_forever()
    except KeyboardInterrupt:
        out("aborted")
        return 130
    finally:
        stopper.restore()
        close()
    return EXIT_OK


# ------------------------------------------------------------------ entry point

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codeagent", description="Offline plan-driven coding agent.")
    p.add_argument("--version", action="version", version=f"codeagent {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", help="config TOML (default: ./codeagent.toml if present)")

    sp = sub.add_parser("template", help="print (or write) a plan template")
    sp.add_argument("--out"); sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("check", help="validate a plan without running it"); common(sp)
    sp.add_argument("plan")
    sp = sub.add_parser("doctor", help="check model server, model, git and system resources"); common(sp)
    sp = sub.add_parser("smoke", help="test whether the model follows the output format"); common(sp)
    sp.add_argument("--tries", type=int, default=5)
    sp = sub.add_parser("bench", help="raw model test: can the model write small functions at all?"); common(sp)
    sp.add_argument("--tries", type=int, default=5, help="samples per problem (default 5)")
    sp.add_argument("--temperature", type=float, help="sampling temperature (default 0.4)")
    sp.add_argument("--only", nargs="+", metavar="PROBLEM", help="run only these problems")
    sp = sub.add_parser("run", help="build a plan"); common(sp)
    sp.add_argument("plan"); sp.add_argument("--workspace"); sp.add_argument("--dry-run", action="store_true")
    sp = sub.add_parser("daemon", help="watch an inbox folder and build plans 24/7"); common(sp)
    sp.add_argument("--inbox", required=True); sp.add_argument("--once", action="store_true")
    return p


COMMANDS = {"template": cmd_template, "check": cmd_check, "doctor": cmd_doctor,
            "smoke": cmd_smoke, "bench": cmd_bench, "run": cmd_run, "daemon": cmd_daemon}


def _silence_stdout() -> None:
    """After a closed pipe, stop Python complaining again while it flushes at exit."""
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (OSError, ValueError):
        pass


def main(argv=None, *, llm_factory=make_client, guard_factory=None, out=print) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, out, llm_factory=llm_factory, guard_factory=guard_factory)
    except (ConfigError, PlanError) as e:
        out(f"error: {e}")
        return EXIT_USAGE
    except BrokenPipeError:                      # e.g. `codeagent template | head`
        return EXIT_OK


def entry() -> None:
    """Console-script entry point (the `codeagent` command)."""
    code = main()
    try:
        sys.stdout.flush()
    except BrokenPipeError:                      # the reader (e.g. `head`) went away
        _silence_stdout()
    sys.exit(code)


if __name__ == "__main__":
    entry()
