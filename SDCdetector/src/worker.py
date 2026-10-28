"""Connect one experiment rank to the controller, then invoke the existing runner. GPT!"""
import os
import socket
import sys

from reporting import Reporter, compact


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.connect(os.environ["EXPERIMENT_CONTROLLER_SOCKET"])
        with channel.makefile("w") as stream:
            def send(message):
                stream.write(compact(message) + "\n")
                stream.flush()

            send({"type": "hello", "experiment": args[0],
                  "rank": int(os.environ.get("RANK", 0))})
            reporter = Reporter(os.environ["EXPERIMENT_LOGGING"], sink=send)
            try:
                from run import main as run
                return run(args, reporter=reporter)
            except (Exception, SystemExit) as exc:
                reporter.test("worker", "ERROR", error=f"{type(exc).__name__}: {exc}")
                reporter.finish("ERROR", f"{type(exc).__name__}: {exc}")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
