"""
fetch_data.py - download fresh copies of the three HPO files into data/ (needs internet)
    python fetch_data.py
"""
import urllib.request
from pathlib import Path

BASE = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/"
OUT = Path(__file__).resolve().parent / "data"

for name in ("hp.obo", "phenotype.hpoa", "genes_to_disease.txt"):
    print("downloading", name)
    (OUT / name).write_bytes(urllib.request.urlopen(BASE + name, timeout=120).read())
print("done")
