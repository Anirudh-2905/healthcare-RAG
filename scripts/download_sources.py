from __future__ import annotations

from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION = PROJECT_ROOT / "data" / "source_docs"

SOURCES = {
    "who_healthy_diet_guideline.pdf": (
        "https://iris.who.int/bitstream/handle/10665/379324/9789240101876-eng.pdf"
    ),
    "who_physical_activity_guideline.pdf": (
        "https://iris.who.int/bitstream/handle/10665/336656/9789240015128-eng.pdf"
    ),
    "who_hypertension_guideline.pdf": (
        "https://iris.who.int/bitstream/handle/10665/344424/9789240033986-eng.pdf"
    ),
    "who_tobacco_cessation_guideline.pdf": (
        "https://iris.who.int/bitstream/handle/10665/377825/9789240096431-eng.pdf"
    ),
    "who_oral_health_tobacco_toolkit.pdf": (
        "https://iris.who.int/bitstream/handle/10665/255628/9789241512510-eng.pdf"
    ),
}


def download(filename: str, url: str) -> None:
    target = DESTINATION / filename
    if target.exists():
        print(f"skip {filename}: already present")
        return
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"{filename}: server did not return a PDF")
    target.write_bytes(response.content)
    print(f"saved {filename} ({len(response.content):,} bytes)")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for filename, url in SOURCES.items():
        try:
            download(filename, url)
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"{filename}: {exc}")
    if failures:
        print("\nSome downloads require manual completion:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
