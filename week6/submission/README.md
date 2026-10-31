# Week 6 class task - Syed Sami Ullah (BSAI-7B)

Put this folder next to the course's `data/` and toolkits (inside `week6/` or `week6/assignments/`). No absolute paths; libraries in `requirements.txt`.

Run in this order, from any folder:

1. `python partA_process.py` - reads the 3 raw CSVs + GeoJSON, prints the audit, writes `clean/` (6 CSVs), `partA_audit.txt` and `partA_decisions.md`. Raw files are never written to.
2. `python partB_charts.py` - reads `clean/`, writes `partB_1.png` (trend), `partB_2.png` (composition), `partB_3.png` (relationship).
3. `python partC_figure.py` - reads `clean/` + `regions.geojson`, writes `partC_figure.png` (300 dpi). The note is `partC_note.md`.

Every self-check number in TASKS.md is an `assert`, so a script stops if a number is off. Bonus (Task D2, real data) is in `../partD_bonus/`.
