from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PUBMEDQA_DATASET = "pubmed_qa"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMED_URL_TEMPLATE = "http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"


def normalize_answer(value: Any) -> str:
    return str(value or "").strip().lower()


def clean_pmid(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    match = re.search(r"\d+", str(value or ""))
    return match.group(0) if match else ""


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_pubmedqa_split(split: str = "train"):
    from datasets import load_dataset

    return load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split=split)


def _context_list(sample: dict[str, Any]) -> list[str]:
    context = sample.get("context") or {}
    if not hasattr(context, "get"):
        return []
    contexts = context.get("contexts") or []
    return [clean_text(context_text) for context_text in contexts if clean_text(context_text)]


def _title_from_context(sample: dict[str, Any], pmid: str) -> str:
    context = sample.get("context") or {}
    if hasattr(context, "get"):
        meshes = context.get("meshes") or []
        if meshes:
            return "; ".join(clean_text(mesh) for mesh in meshes[:6] if clean_text(mesh))
    return f"PubMedQA article {pmid}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_faiss_metadata(path: Path, documents: list[dict[str, Any]], source_filename: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["PMID", "Filename", "Index"])
        writer.writeheader()
        for index, document in enumerate(documents):
            writer.writerow({"PMID": document["PMID"], "Filename": source_filename, "Index": index})


def pubmedqa_to_paper_records(
    dataset,
    max_samples: int | None = 1000,
    max_documents: int | None = 1000,
    max_questions: int | None = 250,
    include_maybe: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    documents_by_pmid: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    answer_counts: Counter[str] = Counter()

    allowed_answers = {"yes", "no", "maybe"} if include_maybe else {"yes", "no"}
    for sample_index, sample in enumerate(dataset):
        if max_samples is not None and sample_index >= max_samples:
            break

        answer = normalize_answer(sample.get("final_decision"))
        answer_counts[answer] += 1
        if answer not in allowed_answers:
            continue

        pmid = clean_pmid(sample.get("pubid"))
        contexts = _context_list(sample)
        question_text = clean_text(sample.get("question"))
        if not pmid or not contexts or not question_text:
            continue

        if pmid not in documents_by_pmid:
            if max_documents is not None and len(documents_by_pmid) >= max_documents:
                continue
            documents_by_pmid[pmid] = {
                "PMID": pmid,
                "title": _title_from_context(sample, pmid),
                "content": "\n\n".join(contexts),
                "source_dataset": f"{PUBMEDQA_DATASET}/{PUBMEDQA_CONFIG}",
                "source_split": "train",
            }

        if max_questions is None or len(questions) < max_questions:
            long_answer = clean_text(sample.get("long_answer"))
            questions.append(
                {
                    "id": f"pubmedqa_{pmid}",
                    "type": "yesno",
                    "body": question_text,
                    "documents": [PUBMED_URL_TEMPLATE.format(pmid=pmid)],
                    "exact_answer": answer,
                    "ideal_answer": [long_answer] if long_answer else [answer],
                    "source_dataset": f"{PUBMEDQA_DATASET}/{PUBMEDQA_CONFIG}",
                }
            )

        if max_questions is not None and len(questions) >= max_questions:
            break

    return list(documents_by_pmid.values()), questions, dict(answer_counts)


def materialize_pubmedqa_sample(
    data_root: str | Path,
    corpus_dir: str | Path | None = None,
    bioasq_json_path: str | Path | None = None,
    max_samples: int | None = 1000,
    max_documents: int | None = 1000,
    max_questions: int | None = 250,
    include_maybe: bool = False,
    overwrite: bool = False,
    split: str = "train",
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    corpus_dir = Path(corpus_dir or data_root / "PubMedAbstractsSubset").expanduser().resolve()
    bioasq_json_path = Path(bioasq_json_path or data_root / "bioASQ" / "yesno_questions.json").expanduser().resolve()
    pubmedqa_dir = data_root / "pubmed_qa"
    mongodb_dir = data_root / "mongodb"
    faiss_dir = data_root / "faiss"

    dataset = load_pubmedqa_split(split=split)
    documents, questions, answer_counts = pubmedqa_to_paper_records(
        dataset,
        max_samples=max_samples,
        max_documents=max_documents,
        max_questions=max_questions,
        include_maybe=include_maybe,
    )
    if not documents or not questions:
        raise RuntimeError("PubMedQA loaded, but no usable yes/no records were materialized.")

    corpus_jsonl = corpus_dir / "pubmedqa_sample.jsonl"
    mirror_jsonl = pubmedqa_dir / "pubmedqa_sample.jsonl"
    mongodb_jsonl = mongodb_dir / "pubmedqa_sample.jsonl"
    faiss_metadata_csv = faiss_dir / "pubmedqa_pmids.csv"
    manifest_path = pubmedqa_dir / "materialization_manifest.json"

    if overwrite or not corpus_jsonl.exists():
        _write_jsonl(corpus_jsonl, documents)
    if overwrite or not mirror_jsonl.exists():
        _write_jsonl(mirror_jsonl, documents)
    if overwrite or not mongodb_jsonl.exists():
        _write_jsonl(mongodb_jsonl, documents)
    if overwrite or not faiss_metadata_csv.exists():
        _write_faiss_metadata(faiss_metadata_csv, documents, corpus_jsonl.name)
    if overwrite or not bioasq_json_path.exists():
        _write_json(bioasq_json_path, {"questions": questions})

    manifest = {
        "source_dataset": f"{PUBMEDQA_DATASET}/{PUBMEDQA_CONFIG}",
        "source_split": split,
        "paper_proxy": "Free-tier PubMedQA yes/no subset materialized as PubMed JSONL plus BioASQ-style questions.",
        "documents": len(documents),
        "questions": len(questions),
        "include_maybe": include_maybe,
        "answer_counts_seen": answer_counts,
        "paths": {
            "corpus_jsonl": str(corpus_jsonl),
            "pubmedqa_jsonl": str(mirror_jsonl),
            "mongodb_jsonl": str(mongodb_jsonl),
            "bioasq_yesno_json": str(bioasq_json_path),
            "faiss_pmids_csv": str(faiss_metadata_csv),
            "manifest": str(manifest_path),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize PubMedQA as paper-style PubMed/BioASQ artifacts.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--max-questions", type=int, default=250)
    parser.add_argument("--include-maybe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = materialize_pubmedqa_sample(
        data_root=args.data_root,
        max_samples=args.max_samples,
        max_documents=args.max_documents,
        max_questions=args.max_questions,
        include_maybe=args.include_maybe,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()