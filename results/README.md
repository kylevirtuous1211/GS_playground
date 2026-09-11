# results/

**Tracked.** Every number quoted in `LOG.md`, in `README.md`, in a slide or in a paper comes from here.

```bash
python results/collect.py          # rebuild tables/ from outputs/*/*/metrics.json
python results/collect.py --check  # exit 1 if the tables disagree with outputs/
```

`collect.py` lives next to the CSVs it produces, so a stale table is visible rather than merely wrong.

- `tables/*.csv` - machine-generated. Never hand-edited. An arm that was not run gets a **blank cell**; the row is not dropped and the cell is not filled in.
- `figures/*.png` - the evidence. A figure is committed only alongside the CSV it was drawn from.
