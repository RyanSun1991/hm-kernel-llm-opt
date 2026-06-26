# `.opencode` Harness — Team Presentation

Slide deck explaining the `.opencode/` multi-agent kernel-optimization harness:
the 3 entry routes, the 7-stage gated pipeline, hub-and-spoke delegation, the
two mandatory gates + A/B-on-hardware rule, handoff contract, skill packs, the
research/ideation engine, cross-run memory + idea ledger, compaction safety,
the MCP toolbelt, and how to run a pipeline. Light context on where it sits
inside the broader `hmopt` platform.

## Files

- `opencode_pipeline_team_brief_2026.pptx` — concise editable team brief (14 slides, English, speaker notes embedded)
- `opencode_pipeline_team_brief_2026_notes.md` — extracted speaker notes for the concise team brief
- `opencode_harness_overview.pptx` — the longer editable PowerPoint deck (19 slides, English)
- `opencode_harness_overview.pdf` — flattened PDF for quick sharing/preview of the longer deck

## Decks

Use `opencode_pipeline_team_brief_2026.pptx` for a weekly team presentation. It is a tighter walkthrough of usage + implementation: where `.opencode` sits in the full `hmopt` pipeline, the 7-stage gated flow, hub-and-spoke delegation, state rebuild, A/B validation, memory/artifacts, and how to run the profiles.

The older `opencode_harness_overview.pptx` is still useful as a deeper reference deck.

## Legacy Rebuild

The generator below rebuilds the older 19-slide deck, not the new artifact-tool team brief.

The generator needs `python-pptx`. Using a throwaway venv:

```bash
python3 -m venv /tmp/pptxbuild
/tmp/pptxbuild/bin/python -m pip install python-pptx
cd docs/presentation
/tmp/pptxbuild/bin/python build_deck.py
```

Output: `docs/presentation/opencode_harness_overview.pptx`.

## Re-export the PDF (optional)

If LibreOffice is installed:

```bash
/Applications/LibreOffice.app/Contents/MacOS/soffice \
  --headless --convert-to pdf --outdir . opencode_harness_overview.pptx
```

## Editing content

Slide text is built in `build_deck.py` — each slide is a clearly commented
section (`SLIDE N — …`). Edit the bullet/title strings there and rerun. Visual
styling (colors, fonts, header/footer, card/table/flow helpers) lives in
`deck_lib.py`.
