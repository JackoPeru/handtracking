from pathlib import Path
import runpy


def main():
    runtime_path = Path(__file__).with_name("handtracking_runtime.py")
    runpy.run_path(str(runtime_path), run_name="__main__")


if __name__ == "__main__":
    main()
