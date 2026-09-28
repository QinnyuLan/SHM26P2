"""Score one checkpoint on the frozen 50/41 original-grid official protocol."""
import argparse

from bridge_rgs.official_evaluate import DEFAULT_REFERENCE, evaluate_official


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-checkpoint", help="Fixed rendered-RGB teacher mixture; no external test images")
    parser.add_argument("--reference-manifest", default=str(DEFAULT_REFERENCE),
                        help="Frozen legacy manifest only; exact SHA and 50/41 set are checked")
    parser.add_argument("--workspace-root", default=".",
                        help="Resolve relative checkpoint/data paths here, including when using a source snapshot")
    args = parser.parse_args()
    evaluate_official(args.checkpoint, args.output, args.reference_manifest,
                      workspace_root=args.workspace_root, entrypoint=__file__,
                      teacher_checkpoint=args.teacher_checkpoint)


if __name__ == "__main__":
    main()
