# Data Layout

```text
data/
├── evaluation/
│   └── questions.json
├── source_docs/
│   └── five approved WHO PDF files
├── chroma/          # generated vector index; ignored by Git
└── catalog.db       # generated document catalogue; ignored by Git
```

## `source_docs/` — input evidence

This directory contains the trusted PDF corpus the assistant is allowed to search. These files are inputs, not application code, and are ignored by Git because they can be downloaded again from their publishers.

Download these five documents from their official WHO publication pages and save them using the exact filenames shown:

| Required filename | Official publication |
|---|---|
| `who_healthy_diet_guideline.pdf` | [What are healthy diets?](https://www.who.int/publications/b/76012) |
| `who_physical_activity_guideline.pdf` | [WHO guidelines on physical activity and sedentary behaviour](https://iris.who.int/bitstream/handle/10665/336656/9789240015128-eng.pdf) |
| `who_hypertension_guideline.pdf` | [Pharmacological treatment of hypertension in adults](https://www.who.int/publications/i/item/9789240033986) |
| `who_tobacco_cessation_guideline.pdf` | [Clinical treatment guideline for tobacco cessation in adults](https://www.who.int/publications/b/74755) |
| `who_oral_health_tobacco_toolkit.pdf` | [Toolkit for oral health professionals](https://www.who.int/publications/i/item/9789241512510) |

Each publication page has a download control. Alternatively, run:

```bash
python scripts/download_sources.py
```

If WHO rejects an automated request, download that file in the browser and place it in `data/source_docs/` with the corresponding required filename.

## `evaluation/questions.json` — measurement specification

This JSON file contains 20 test cases:

- 10 answerable questions whose evidence should be present in one document;
- 4 multi-document questions that test retrieval across documents;
- 3 unrelated questions that should produce an insufficient-evidence response;
- 3 safety questions that should be rejected or safely abstained from.

Each object has this structure:

```json
{
  "id": "A01",
  "category": "answerable",
  "question": "How much moderate-intensity physical activity should adults perform each week?",
  "expected_documents": ["who_physical_activity_guideline.pdf"],
  "expected_keywords": ["150", "minutes"],
  "should_abstain": false
}
```

Field meanings:

- `id`: stable identifier for comparing results between runs.
- `category`: factual, multi-document, unanswerable, or safety test group.
- `question`: text sent to the same query pipeline used by the API.
- `expected_documents`: filenames that should appear among the returned citations.
- `expected_keywords`: simple expected terms used for an initial answer-quality check.
- `should_abstain`: whether the system should refuse to make a supported factual claim.

The evaluator calculates retrieval hit@k, citation presence, citation correctness, expected-document coverage, abstention accuracy, keyword coverage, and average latency. Citation correctness measures how many returned source filenames are expected for the case; document coverage measures how much of the expected source set was retrieved. It does not train the model and does not guarantee medical correctness; it is a small engineering regression suite.

## Generated data

After ingestion, Chroma writes embeddings under `data/chroma/`, while SQLite writes document metadata to `data/catalog.db`. Both survive an application restart but can be rebuilt from `source_docs/`.
