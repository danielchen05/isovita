"""Command-line adapter for AnnData preparation."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence
from .preprocessing import (
    ad,
    read_table,
    normalize_separator,
    resolve_sample_id_parser,
    prepare_isovita_inputs,
    save_prepared_inputs,
)


def parse_filter_arguments(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated ``--filter column=value`` CLI arguments."""
    filters: Dict[str, str] = {}

    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid filter {item!r}. Expected COLUMN=VALUE.")

        column, value = item.split("=", 1)
        column = column.strip()
        value = value.strip()

        if not column:
            raise ValueError(f"Invalid filter {item!r}: empty column name.")

        filters[column] = value

    return filters


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one standardized input set for IsoVita.")

    parser.add_argument(
        "--adata",
        required=True,
        help="Path to transcript-level AnnData .h5ad file.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to sample metadata table.",
    )
    parser.add_argument(
        "--tx2gene",
        required=True,
        help="Path to transcript-to-gene mapping table.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for prepared IsoVita inputs.",
    )

    parser.add_argument(
        "--metadata-sep",
        default="tab",
        choices=["tab", "tsv", "comma", "csv"],
        help="Metadata delimiter. Default: tab.",
    )
    parser.add_argument(
        "--tx2gene-sep",
        default="comma",
        choices=["tab", "tsv", "comma", "csv"],
        help="tx2gene delimiter. Default: comma.",
    )

    parser.add_argument(
        "--metadata-sample-col",
        default="sample_id",
        help="Metadata sample-ID column. Default: sample_id.",
    )
    parser.add_argument(
        "--age-col",
        default="age",
        help="Metadata age column. Default: age.",
    )

    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=("Exact-match metadata filter COLUMN=VALUE. May be supplied multiple times."),
    )

    parser.add_argument(
        "--abundance-key",
        default="abundance",
        help="AnnData.obsm key containing sample x transcript abundance.",
    )

    parser.add_argument(
        "--sample-id-mode",
        default="identity",
        choices=["identity", "underscore_prefix"],
        help=("How to convert AnnData obs_names into metadata sample IDs. Default: identity."),
    )
    parser.add_argument(
        "--sample-id-parts",
        type=int,
        default=2,
        help=(
            "Number of underscore-delimited basename components retained "
            "when --sample-id-mode=underscore_prefix. Default: 2."
        ),
    )

    parser.add_argument(
        "--tx2gene-transcript-col",
        default=None,
        help=("Transcript column in tx2gene. If omitted, common names are auto-detected."),
    )
    parser.add_argument(
        "--tx2gene-gene-col",
        default=None,
        help=("Gene column in tx2gene. If omitted, common names are auto-detected."),
    )

    parser.add_argument(
        "--include-prenatal",
        action="store_true",
        help=(
            "Retain negative ages. By default preparation restricts to age >= 0 "
            "so the core receives postnatal-only metadata."
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    metadata_sep = normalize_separator(args.metadata_sep)
    tx2gene_sep = normalize_separator(args.tx2gene_sep)

    metadata = read_table(args.metadata, sep=metadata_sep)
    tx2gene = read_table(args.tx2gene, sep=tx2gene_sep)
    if ad is None:
        raise ImportError("The CLI/AnnData adapter requires the optional 'anndata' package.")
    adata = ad.read_h5ad(args.adata)

    filters = parse_filter_arguments(args.filter)
    sample_parser = resolve_sample_id_parser(
        args.sample_id_mode,
        n_parts=args.sample_id_parts,
    )

    result = prepare_isovita_inputs(
        adata=adata,
        metadata=metadata,
        tx2gene=tx2gene,
        metadata_sample_col=args.metadata_sample_col,
        age_col=args.age_col,
        metadata_filters=filters,
        abundance_key=args.abundance_key,
        sample_id_parser=sample_parser,
        tx2gene_transcript_col=args.tx2gene_transcript_col,
        tx2gene_gene_col=args.tx2gene_gene_col,
        postnatal_only=not args.include_prenatal,
    )

    save_prepared_inputs(result, args.output_dir)

    print("\nIsoVita input preparation complete.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print("\nPreparation summary:")
    print(result.summary.to_string(index=False))
