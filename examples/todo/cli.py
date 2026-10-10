"""Hand-written glue: the command line. All the logic lives in the functions the agent built."""
import argparse

from . import add_task, complete_task, filter_tasks, format_task, remove_task, summarize
from .store import load_tasks, save_tasks


def build_parser():
    parser = argparse.ArgumentParser(prog="todo", description="A tiny todo list")
    parser.add_argument("--file", default="tasks.json", help="where the tasks are saved (default: tasks.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("add", help="add a task").add_argument("text")
    sp = sub.add_parser("list", help="show tasks")
    sp.add_argument("--status", default="all", choices=["all", "open", "done"])
    sub.add_parser("done", help="mark a task as done").add_argument("id", type=int)
    sub.add_parser("rm", help="remove a task").add_argument("id", type=int)
    return parser


def main(argv=None, out=print):
    args = build_parser().parse_args(argv)
    tasks = load_tasks(args.file)
    try:
        if args.command == "add":
            tasks = add_task(tasks, args.text)
            save_tasks(args.file, tasks)
            out("added " + format_task(tasks[-1]))
        elif args.command == "list":
            for task in filter_tasks(tasks, args.status):
                out(format_task(task))
            out(summarize(tasks))
        elif args.command == "done":
            tasks = complete_task(tasks, args.id)
            save_tasks(args.file, tasks)
            out(f"task {args.id} is done")
        elif args.command == "rm":
            tasks = remove_task(tasks, args.id)
            save_tasks(args.file, tasks)
            out(f"task {args.id} removed")
    except KeyError:
        out(f"error: there is no task with id {args.id}")
        return 1
    except ValueError:
        out("error: the task text cannot be empty")
        return 1
    return 0
