"""Command-line interface.

    python -m emotion_reranking run  [--data PATH] [--output DIR] [--queries MODE] [--n-docs N]
    python -m emotion_reranking plot [--results DIR]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from emotion_reranking.config import QUERY_MODES, ExperimentConfig


def _run(args: argparse.Namespace) -> None:
    from emotion_reranking.pipeline import ExperimentPipeline

    config = ExperimentConfig(data_path=args.data, output_dir=args.output,
                              query_mode=args.queries, n_docs=args.n_docs)
    ExperimentPipeline(config, make_figures=not args.no_figures).run()


def _plot(args: argparse.Namespace) -> None:
    from emotion_reranking.pipeline import ExperimentResults
    from emotion_reranking.visualization import generate_figures

    results = ExperimentResults.load(args.results)
    out_dir = args.figures or Path(args.results) / "figures"
    for path in generate_figures(results.metrics, results.significance, out_dir):
        logging.info("Saved %s", path)


def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(prog="python -m emotion_reranking",
                                     description="Emotion-aware re-ranking and rank fusion of Amazon reviews.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full experiment")
    run.add_argument("--data", type=Path, default=defaults.data_path,
                     help="path to the Amazon Reviews CSV (downloaded from Kaggle if missing)")
    run.add_argument("--output", type=Path, default=defaults.output_dir, help="output directory")
    run.add_argument("--queries", choices=QUERY_MODES, default=defaults.query_mode, help="query set")
    run.add_argument("--n-docs", type=int, default=defaults.n_docs, help="corpus size after sampling")
    run.add_argument("--no-figures", action="store_true", help="skip figure generation")
    run.set_defaults(func=_run)

    plot = sub.add_parser("plot", help="regenerate figures from saved results")
    plot.add_argument("--results", type=Path, default=defaults.output_dir, help="directory with result CSVs")
    plot.add_argument("--figures", type=Path, default=None, help="output directory (default: RESULTS/figures)")
    plot.set_defaults(func=_plot)
    return parser


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
