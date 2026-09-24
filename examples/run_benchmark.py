"""Run a small, reproducible Kaggle metadata acquisition benchmark."""

import json
import shutil

from kagglekit.acquisition_benchmark import benchmark


def main() -> None:
    methods = ["rest-sequential", "rest-concurrent"]
    if shutil.which("kaggle"):
        methods.append("kaggle-cli")
    report = benchmark(
        ["uciml/iris", "kaggle/sf-salaries"],
        methods,
        workers=4,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

