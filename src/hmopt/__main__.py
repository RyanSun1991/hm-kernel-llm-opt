"""Allow `python -m hmopt` as an alias for the `hmopt` console script.

People naturally try `python -m hmopt <cmd>` before discovering the installed
entry point; without this file that fails with a confusing "No module named
hmopt" -style error even on a correctly installed package.
"""
from hmopt.cli import app

if __name__ == "__main__":
    app()
