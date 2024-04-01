from shutil import get_terminal_size as tsize
import math



_print = print

# does not work across multiprocessing threads. Oh well.
lastProgress = 0
lastProgressName = None

def print(text: str):
    global lastProgress, lastProgressName
    tsiz = tsize()[0]
    _print(f"\r{text}{' ' * (tsiz*math.ceil(len(text)/tsiz)-len(text) - 1)}\n", end="", flush=lastProgressName is None)
    if lastProgressName is not None:
        printProgress(lastProgressName, lastProgress)


def printProgress(name: str, progress: float, done=False):
    global lastProgress, lastProgressName
    tsiz = tsize()[0] - 15
    filled = int(progress * tsiz)
    _print(f"\r{name:4} {progress * 100:5.1f}% [{'=' * filled}{' ' * (tsiz - filled)}]" + ('\n' if done else ''), end="", flush=True)
    if done:
        lastProgressName = None
    else:
        lastProgress = progress
        lastProgressName = name