"""Run a fully recorded teacher configuration using the project's uv environment."""

import argparse
import json
from pathlib import Path

from bridge_rgs.teacher import TeacherConfig, train_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    result = train_teacher(
        settings["manifest"],
        settings["model_dir"],
        settings["output_dir"],
        TeacherConfig(**settings["config"]),
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
