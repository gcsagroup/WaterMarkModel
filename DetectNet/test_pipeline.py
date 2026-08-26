"""Run the configured test directory and create test/result.json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import DetectionPipeline
    from settings import DEFAULT_CONFIG
else:
    from .pipeline import DetectionPipeline
    from .settings import DEFAULT_CONFIG


def run_test(config_path: str | Path = DEFAULT_CONFIG) -> list[object]:
    pipeline = DetectionPipeline(config_path)
    results = pipeline.test()
    print(
        f"测试完成: total={len(results)}, "
        f"watermark_candidates={sum(result.has_watermark for result in results)}, "
        f"errors={sum(result.error is not None for result in results)}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="运行Detect测试目录")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run_test(args.config)


if __name__ == "__main__":
    main()
