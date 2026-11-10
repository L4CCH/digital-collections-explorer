---
name: dce-setup
description: Turn a folder, S3 bucket, or parquet manifest of images, PDFs or 3D models into a searchable collection site with Digital Collections Explorer. Use when someone wants to explore, search, index, or publish a digital collection, including one with no metadata, no catalog, or messy file names, or one whose images are served by an IIIF image server. Inspects the collection, says what is and is not possible, asks only what a person must decide, builds the index, and starts the site.
license: MIT
compatibility: Run from the root of the digital-collections-explorer repository. Needs Python 3.10+, Node.js 20.19+ and a shell, and network access for S3, web or IIIF sources.
metadata:
  author: digital-collections-explorer
  version: "2.0"
---

# Set up a searchable collection

The person you are helping is probably a curator, librarian, archivist or
researcher, not an engineer. They know their collection. They do not know what a
GPU or an embedding is, and they should never need to. Take them from "the
images are over there" to a working search site, making every technical decision
yourself and asking only what a machine cannot find out.

## Ground rules

1. **Measure, do not ask.** Hardware, disk space, file counts, formats, download
   speed and available metadata are measured by the commands below.
2. **Ask only what a person must decide:** policy and intent, such as whether data
   may leave their machines and what the site keeps of images their source can
   show. Give each question a recommended answer and a one-sentence reason.
3. **Say what is not possible, why, and what would change it.** Do not quietly
   leave something out, and do not promise it.
4. **Never replace or mix collections.** Give every source a folder of its own
   that does not exist yet. Indexing into the folder of another collection adds
   to it. If `ingest` refuses because the folder holds an index, choose another
   `--data-dir`, and pass `--overwrite` only when the person asks to replace that
   index.
5. **Never spend money without consent.** Before creating any cloud resource,
   state the monthly cost and wait for a clear yes.
6. **Keep three kinds of information apart:** the institution's catalog, what is
   read from the files (dates in photo files, file names, folders), and machine
   guesses. Never present a guess as catalog fact.
7. **Use plain language.** Say "analyzing the images", not "computing
   embeddings", and "memory", not "RAM". Give times and sizes in minutes and MB.
   Never show internal field names, JSON, process ids or raw errors: translate
   an error into what happened, such as "2 files are damaged and could not be
   opened".
8. **Report measurements, not guesses.** The profile's `sizing` is an estimate
   made before indexing. Afterwards, report only what the ingest summary
   measured.

## Python

```bash
PY=venv/bin/python; [ -x "$PY" ] || PY=python
```

If imports fail: `python -m venv venv && venv/bin/pip install -r requirements.txt`.

## Step 1. Look at the machine

```bash
$PY -m src.environment
```

Keep the JSON to yourself. You need `device`, `memory_bytes`, `free_disk_bytes`,
`aws_credentials_found`, `frontend_built`, `frontend_media` and
`python_packages`.

## Step 2. Look at the collection

```bash
$PY -m src.profiling /path/to/folder --json /tmp/profile.json
$PY -m src.profiling s3://bucket/prefix --anonymous --json /tmp/profile.json
$PY -m src.profiling manifest.parquet --json /tmp/profile.json
```

Use `--anonymous` for a public bucket, and leave it off when
`aws_credentials_found` is true and the bucket is private. A manifest names each
image in a column such as `image_url`, or only by its IIIF service in a column
such as `iiif`, with any catalog fields beside it, and a `group` column when
the site holds several institutions or museums. [Mapping your data](https://l4cch.github.io/dce-website/docs/mapping-your-data/)
lists the columns the site reads and shows how to make a manifest from a
spreadsheet, a collection system's export or IIIF manifests. Read the printed report and
`/tmp/profile.json`.

**If the report warns that image files could not be read**, fix that first. The
usual cause is a URL column that points to web pages rather than files: ask
where the files are stored and rerun with `--fetch-via s3://bucket`.

`media` in the profile lists what the collection holds besides images: `pdf`,
`3d`, and `zoom` for records that name an IIIF image service. Build the site for
exactly those, unless `frontend_built` is true and `frontend_media` matches:

```bash
npm install && npm run setup -- --with=MEDIA
```

`MEDIA` is the list joined by commas, or `none`. Then install what the list
needs, unless `python_packages` says it is there:
`$PY -m pip install -r requirements-pdf.txt` for `pdf`, and
`$PY -m pip install -r requirements-3d.txt` for `3d`.

## Step 3. Tell the person what you found

A few short paragraphs, leading with whatever would surprise them most:

1. What is there: how many images, PDFs and 3D models, how large, and anything
   unexpected such as images with no catalog record or damaged files. Catalog
   columns in a manifest are used. A spreadsheet lying next to the images is
   not read yet: say so, and mention it if it looks empty.
2. What they will be able to do: the report's "The site will" list, which
   follows the rules the site itself uses. "What the site does with each
   column" says how each column is read. A column whose name the site does not
   know is read as notes; renamed, such as `made_by` to `creator`, it becomes a
   name to search and filter by.
3. What they will not be able to do, why, and what would change that.
4. What it will cost: time, disk space, and whether this machine can handle it.
   Compare `sizing` with `free_disk_bytes` and `memory_bytes`. Its disk estimate
   assumes copies are kept, so it is smaller with `thumbnails` and close to
   nothing with `link`.

Dates and places read from photo files record when and where the picture was
taken. For scanned documents or photographed objects that is the digitization,
rarely what a curator means by "date". Say which applies, and ask if unsure.

## Step 4. Ask the decisions

The profile's `decisions` hold the questions, each with `options`, a
`recommended` answer and `why`. Two are settled already, so tell the person
instead of asking: images with no catalog record are always indexed, and
visitors can hide them with "With catalog records only"; files that are not
images are always left out.

- **Small local job** (under five minutes, source on this machine, no cost): go
  ahead with the recommendations, then say what you chose so they can object.
  Skip the question about the cloud.
- **Anything larger, remote, or costing money:** ask the questions together, in
  plain language, with the recommendation and reason. Use a structured question
  tool if you have one. "Whatever you recommend" is an answer.

If indexing will take more than an hour, also ask whether to index a few hundred
images first, so they see the result in minutes.

The `images` decision appears when the source can show the images itself: it
names an IIIF service, web addresses of images, or thumbnails. It sets
`--images`, which is `copy` when the profile has no such decision:

| Answer | The site keeps | Visitors' browsers load |
| --- | --- | --- |
| `copy` | copies of every image | everything from this site, except deep zoom from an IIIF server |
| `thumbnails` | small previews | previews from this site, larger views and zoom from the source |
| `link` | nothing, unless the source has neither an IIIF service nor its own thumbnails, which keeps a preview | everything else from the source |

Images the source cannot show, such as TIFF files, are copied whatever the
answer. Linking depends on the source staying up, fast and at the same
addresses, and on it letting other sites show its images. Say so when you
recommend it.

## Step 5. Build the index

If `media` includes `3d`, first draw the models as in
[references/3d-models.md](references/3d-models.md), then come back here.

Give each collection a new folder under `data/collections/`:

```bash
$PY -m src.ingest SOURCE --data-dir data/collections/NAME --images MODE \
  --estimated-seconds SECONDS --json data/collections/NAME/ingest_summary.json
```

`SECONDS` is `sizing.embed_seconds.estimated` from the profile, or `compute` when
the download speed could not be measured, so the report can compare it with
what happened. `SOURCE`, `--anonymous` and `--fetch-via` are as in step 2.
`--limit N` indexes the first N files in name order for a sample, which may all
come from one folder: say so.

- Run anything longer than a few minutes in the background, and check it with
  `$PY -m src.ingest --status --data-dir data/collections/NAME`.
- It can be stopped at any time. The same command continues where it left off.
  `--retry-failed` tries failed images again.
- Originals are never stored. An image with an IIIF service is requested from
  it at the size needed, not downloaded whole.
- A few failures are normal. Report the count and the reasons from `problems`,
  and investigate only above a few percent.
- Each page of a PDF becomes an image of its document, and the document's page
  shows them in order.
- `$PY -m src.ingest --report --data-dir data/collections/NAME` describes a run
  in plain words, also long after it finished.

## Step 6. Start the site

```bash
PORT=8000
DCE_DATA_DIR=data/collections/NAME DCE_PORT=$PORT $PY -m src.backend.main
```

If the port is taken, another collection is probably served there: leave it
running and use the next free port. Never stop a server you did not start
without asking. Run it in the background, then check that it is up and serving
this collection:

```bash
curl -s http://localhost:$PORT/api/health
curl -s "http://localhost:$PORT/api/search/text?query=a%20portrait&limit=3"
```

`collection` must equal `NAME`, `items` must equal `indexed` in the summary, and
the search must return results.

Without a `collection.json` in `data/collections/NAME/`, the site shows a
generic title and no suggested searches. Write one with the person from
`collection.example.json`: a title, one or two sentences of introduction, the
institution, the license, and six to eight searches you have run and that return
good results. With a `group` column, add `groups`, giving each group its
`label`, `institution` and `license`; an object's page shows its group's
([Collection format](https://l4cch.github.io/dce-website/docs/collection-format/#several-institutions-in-one-site)).
The site reads it on every visit, so no restart is needed.

## Step 7. Tell the person how it went

From the ingest summary and nothing else:

1. How long indexing took, from `run`, against your estimate.
2. How many images were indexed, skipped and failed, naming the failed files
   from `run.failed_files`.
3. How much disk the collection uses: `disk_bytes_measured.total`, plus the
   `models3d` folder for 3D models, which it leaves out.
4. What it cost and costs from now on: nothing extra on their own machine; on
   AWS, what `deploy/aws/status.sh NAME` reports, passing on any WARNING about a
   large machine still running first.

Give them the address and two or three searches suited to the collection:
a description of what a picture shows and, where there is a catalog, a name or
a date from it.

## More

- Something went wrong: [references/troubleshooting.md](references/troubleshooting.md).
- Putting the site on a server or AWS, opening it to the public, or letting AI
  assistants search it, only when the person asks:
  [references/deploy.md](references/deploy.md).

Be honest about what is not built yet: one-step publishing to Hugging Face
Spaces or to a cloud other than AWS (any Linux machine works), and a catalog
kept apart from the manifest, in a second table or a spreadsheet beside the
images: its columns have to be in the manifest itself.
