#!/usr/bin/env python3
"""Write the on-device semantic search parity fixture.

Reference vectors come from desktop ONNX Runtime with the Hugging Face
tokenizer for the pinned multilingual-e5-small revision, so the device test
checks the Kotlin tokenizer, the Android runtime and the ranking together.

Requires: onnxruntime, tokenizers, numpy. Run from the project root:
  python scripts/generate-semantic-search-device-fixture.py --tokenizer-json PATH
where PATH is onnx/tokenizer.json from intfloat/multilingual-e5-small at
614241f622f53c4eeff9890bdc4f31cfecc418b3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "app/src/main/generatedEmbeddingAssets/embedding/multilingual_e5_small_qint8.onnx"
OUTPUT = ROOT / "app/src/androidTest/assets/semantic-search/device-reference.json"
# Synthetic transcripts written for this fixture: no recording, no real person.
# Keep it that way; this file and its output are public.
SAMPLES = {
    "synthetic-german-english-planning": [
        "Also ich wollte kurz festhalten, was wir heute im Team besprochen haben.",
        "The release is planned for the second week of March.",
        "Bis dahin müssen die Tests auf dem Pixel und auf dem Tablet durchlaufen.",
        "Mara kümmert sich um die Übersetzungen ins Spanische und Französische.",
        "We still need someone to review the database migration.",
        "Das Budget für die neuen Testgeräte ist noch nicht freigegeben.",
        "I will ask finance again on Monday.",
        "Die Akkulaufzeit war im letzten Test deutlich schlechter als erwartet.",
        "Probably the background sync wakes the phone up too often.",
        "Wir treffen uns am Donnerstag um zehn Uhr wieder.",
        "Until then everybody writes down their open questions.",
        "Und bitte denkt daran, die Stundenzettel abzugeben.",
    ],
    "synthetic-german-weekend": [
        "Am Samstag sind wir früh losgefahren, weil es später regnen sollte.",
        "Die Wanderung ging zuerst durch den Wald und dann am See entlang.",
        "Oben auf dem Gipfel war es ziemlich windig und kalt.",
        "Wir haben Brote, Äpfel und eine Thermoskanne Tee dabeigehabt.",
        "Auf dem Rückweg hat es dann tatsächlich angefangen zu gewittern.",
        "Der Zug nach Hause hatte vierzig Minuten Verspätung.",
        "Abends haben wir Nudeln mit Tomatensoße gekocht.",
        "Am Sonntag habe ich nur gelesen und die Wäsche gemacht.",
        "Nächstes Mal möchte ich gern mit dem Fahrrad fahren.",
        "Dafür muss ich aber erst die Bremsen reparieren lassen.",
    ],
    "synthetic-english-short": [
        "The ferry leaves at half past six.",
    ],
    "synthetic-english-story": [
        "The lighthouse keeper noticed the lamp flickering just after midnight.",
        "She climbed the stairs with a wrench and a tin of oil.",
        "Far out on the water a fishing boat was signalling for help.",
        "Its engine had failed and the tide was pulling it toward the rocks.",
        "She called the harbour on the old radio.",
        "Nobody answered at first.",
        "Then a sleepy voice promised to send the rescue crew.",
        "Until they arrived she kept the lamp turning by hand.",
        "By sunrise the boat was tied up safely at the pier.",
        "The fishermen brought her a basket of mackerel as thanks.",
        "She fried two of them for breakfast and gave the rest to the cat.",
        "The broken gear was replaced the following week.",
    ],
    "synthetic-english-appointments": [
        "I called the dentist this morning.",
        "They can see me on Tuesday at nine.",
        "After that I have to pick up the parcel from the post office.",
        "The landlord wants the rent transferred before the first.",
        "My sister's birthday is on the weekend.",
        "I still need a present for her.",
        "Maybe a record, she likes old jazz.",
        "The car needs new tyres before winter.",
        "And I should finally book the flights to Lisbon.",
    ],
}
# Must match SemanticTranscriptRanking.HUB_ANCHOR_QUERIES.
HUB_ANCHORS = [
    "weather", "money", "family", "work", "food", "travel", "health", "music",
    "Wetter", "Arbeit", "Familie", "Essen", "tiempo", "trabajo", "météo", "travail",
]
MIN_SCORE = 0.03
MAX_HITS = 5
QUERIES = {
    "synthetic-german-english-planning": ["battery drain", "Übersetzung", "money for hardware", "next meeting", "Kochrezept"],
    "synthetic-german-weekend": ["thunderstorm", "Essen kochen", "train delay", "football match"],
    "synthetic-english-story": ["boat in danger", "Seenotrettung", "fish as a gift", "cooking dinner"],
    "synthetic-english-appointments": ["doctor appointment", "Geschenk", "travel plans"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    arguments = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer_json))
    session = ort.InferenceSession(str(MODEL))

    def embed(text: str) -> np.ndarray:
        ids = np.array([tokenizer.encode(text).ids[:256]], dtype=np.int64)
        hidden = session.run(None, {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        })[0][0]
        vector = hidden.mean(0)
        return vector / np.linalg.norm(vector)

    hub = np.mean([embed("query: " + anchor) for anchor in HUB_ANCHORS], axis=0)
    samples = []
    for sample_id, segments in SAMPLES.items():
        vectors = np.array([embed("passage: " + segment) for segment in segments])
        queries = []
        for query in QUERIES.get(sample_id, []):
            scores = vectors @ embed("query: " + query) - vectors @ hub
            order = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] >= MIN_SCORE][:MAX_HITS]
            queries.append({"query": query, "hits": order, "scores": [round(float(scores[i]), 5) for i in order]})
        samples.append({
            "id": sample_id,
            "segments": segments,
            "vectors": [[round(float(x), 6) for x in vector] for vector in vectors],
            "queries": queries,
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "model": "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 model_qint8_avx512_vnni.onnx",
        "minScore": MIN_SCORE,
        "samples": samples,
    }, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {sum(len(s['segments']) for s in samples)} segments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
