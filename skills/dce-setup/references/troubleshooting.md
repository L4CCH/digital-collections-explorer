# Troubleshooting

## "already holds an index that this tool did not create"

`ingest` found a collection in the target folder, an `index_info.json` or an
`embeddings.pt`, without its own state file. Someone built it another way. Use a different `--data-dir`. Only pass
`--overwrite` when the person has explicitly said to replace that index.

## "holds an index in the old layout"

The collection was indexed by v2.0.0-beta.2 or earlier, into `embeddings.pt`,
`item_ids.pt` and `metadata.json`, which the site no longer reads. If the folder has an `ingest_state.sqlite`, run the same `src.ingest`
command again: every file is already indexed, so it only writes the collection
anew. Otherwise it was built another way; serve it with v2.0.0-beta.2, or index
its source again into a new `--data-dir`.

## "was built with ..." or "was indexed with ..., but config.json is now set to ..."

An index can only be searched, or added to, with the model that built it. The
site or `ingest` says so when `config.json` names a different model from the one
the collection was indexed with, for example after the project's default model
changed.

Do not edit the index. Either point `config.json` back at the model named in the
message, or index the collection again into a new `--data-dir` with the current
model. Re-indexing needs the source images again, so say how long it will take
before starting.

## The report says image files could not be read

The manifest's URL column does not return image data. Open one URL to confirm:
it is usually a landing page. Find out where the files are actually stored and
pass `--fetch-via s3://bucket`. The object key is taken from the URL path.

## Download speed is much slower than expected

The `transfer` time in the profile is measured from this machine to the source.
Concurrency rarely helps when the link itself is the limit. For large remote
collections the realistic options are: run overnight (the run can be resumed),
index a sample first, or run on a cloud machine in the same region as the data.
The last option costs money and may conflict with data residency rules, so ask.

## Many images from an IIIF or web server fail to download

Each failure says why, in `problems` and in `--report`. Ingest has already
tried every request again for about half a minute.

- `the server could not be reached` or `did not answer`: this machine's network,
  or the server, was down for longer than that. Check the network, then run the
  same command with `--retry-failed`.
- `the server answered 429` or `503`: the server limits how fast one machine can
  ask, as many institutions' servers do. Wait a while, then run the same command
  with `--retry-failed` and `--requests-per-second`, such as 2.
- `the server answered 404`: the source no longer has those images, or the
  manifest names them wrongly. Open one of the addresses in a browser, and
  report how many.

## Many images fail with "decode" errors

Check `problems` in the ingest summary. `UnidentifiedImageError` means the file
is not a real image despite its extension, a problem in the source data rather
than in the tool. Report the count and a few file names so the person can look
into them. An incomplete file is still indexed, with the missing part gray.

## Images larger than 200 MB are skipped

This is deliberate, to protect memory. Raise it with `--max-file-mb` only when
`memory_bytes` from the environment check is comfortably larger than the files.

## The site starts but shows no results

Confirm the server was started with `DCE_DATA_DIR` pointing at the same folder
that was given to `--data-dir`. Without it the server reads the folders named in
`config.json`.

## Port 8000 is already in use

The server refuses to start and suggests another port. Another collection is
probably being served there: `curl -s http://localhost:8000/api/health` shows
which one. Leave it running and start this collection with `DCE_PORT=8001` (or
the next free port). Only stop the other server if the person asks you to.

## `ModuleNotFoundError` for boto3 or duckdb

Run `venv/bin/pip install -r requirements.txt`. These two packages are needed
for S3 and parquet sources.
