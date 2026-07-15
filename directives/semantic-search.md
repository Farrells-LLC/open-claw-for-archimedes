# Semantic Search

Tags: #directive #search #memory

Use semantic search when exact filename or text search is not enough and loading the whole workspace would waste context.

## QMD Setup

If QMD is installed, index this workspace:

```bash
qmd collection add /home/dash/.openclaw/workspace --name openclaw-workspace
qmd embed
```

Search examples:

```bash
mcporter call qmd.vsearch query="what did we decide about memory maintenance"
mcporter call qmd.query query="Archimedes report QA notes"
```

Run `qmd embed` again after adding significant new memory or second-brain files.

## Fallback

When QMD is unavailable, use `rg` first for exact text and filenames, then read only the relevant files.

