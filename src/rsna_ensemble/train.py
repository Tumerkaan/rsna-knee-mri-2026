"""Explicit, deadline-supervised local training entry point; never dispatches jobs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def terminate_tree(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def main():
    began = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=["abig","t1","f","g","z"])
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wall-seconds", type=int, required=True)
    p.add_argument("--execute", action="store_true", help="Required to train; omission only validates arguments")
    p.add_argument("--child-start", type=float, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({"status":"NOT_RUN", "family":args.family, "fold":args.fold}))
        return
    if args.child_start is None:
        if args.output.exists():
            p.error("output must be a new directory")
        command = [sys.executable, "-m", "rsna_ensemble.train", *sys.argv[1:],
                   "--child-start", str(began)]
        child = subprocess.Popen(command, stderr=subprocess.DEVNULL)
        timed_out = False
        try:
            code = child.wait(timeout=max(1, args.wall_seconds-15))
        except subprocess.TimeoutExpired:
            timed_out, code = True, None
        finally:
            terminate_tree(child.pid)
        args.output.mkdir(parents=True, exist_ok=True)
        receipt = {"exit_code":code, "deadline_reached":timed_out,
                   "wall_seconds":time.monotonic()-began, "automatic_retry":False,
                   "source_error_records":[str(p.relative_to(args.output)).replace("\\","/")
                       for p in args.output.rglob("*error*.json")],
                   "traceback_unavailable_reason":"supervisor termination" if timed_out else None}
        (args.output/"supervisor.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise SystemExit(0 if code == 0 and not timed_out else 1)
    try:
        from .local_runtime import configure
        module = configure(args.family, args.fold, args.bundle, args.output,
                           args.child_start, args.wall_seconds)
        if args.family == "f":
            module._train()
        else:
            module.train_fold(args.fold)
    except BaseException as exc:
        from .local_runtime import save_failure
        args.output.mkdir(parents=True, exist_ok=True)
        save_failure(exc, "TRAIN_ENTRY", args.output)
        print(json.dumps({"status":"STOP", "error_record":"source-error.json"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
