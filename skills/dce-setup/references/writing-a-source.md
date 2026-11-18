# Writing a source

Use this when the collection is not a folder, a bucket, or a manifest: its
records and images are behind an API, in a collection management system that
exports nothing usable, or on an aggregator. A source is a short program that
reads them and writes a manifest. The steps in SKILL.md then profile and index
that manifest like any other. [Writing a source](https://l4cch.github.io/dce-website/docs/sources/)
on the website has a complete example.

## 1. Find out where the records are

- Ask the person for the address of the API and of its documentation, unless
  they gave them.
- If the API needs a key, ask the person to set it in an environment variable,
  such as `MUSEUM_API_KEY`, and read it from there. Never ask for the key
  itself, and never write it into a file.
- Read the documentation, then fetch one page with `curl -A NAME-source/1.0` and
  look at a record. Find the fields that hold the image, as an IIIF image
  service or an image's address; the record's page at the institution; and the
  title, dates, names, places, kinds of object, and materials.
- Read the API's terms of use too, before the program runs in full. Note how
  many requests they allow a second and a day, how long what the API returns
  may be kept, whether its images may be copied or only shown from its own
  servers, and how to credit it. Without such terms, wait a second between
  requests.

## 2. Write the program

Write `data/sources/NAME/source.py`, where `NAME` is the collection's folder
name, so that it writes `manifest.parquet` beside itself. Git ignores
`data/sources/`. Follow the example on the website:

- Name the program in a `User-Agent` header, since many APIs refuse the default
  one.
- Page through the records until a page comes back empty, and try a failed
  request again after a pause.
- Send one request at a time, at the pace the terms allow, and stop before a
  daily limit.
- Ask only for the fields you use, if the API allows it, so that its answers
  stay small.
- Skip records without an image, and read every field with `.get`.
- Put an IIIF image service in `iiif`, or else the image's address in
  `image_url`; the record's id in `object_id`; and its page in `source_url`.
  Give the other fields the names of columns the site knows, such as `title`,
  `date`, `creator`, `place`, `object_type`, and `technique`, from
  [Mapping your data](https://l4cch.github.io/dce-website/docs/mapping-your-data/#how-columns-are-read),
  and join several values with `; `.

## 3. Check it on a sample

The first time, stop the program after a few hundred records, then profile what
it wrote:

```bash
$PY -m src.profiling data/sources/NAME/manifest.parquet
```

Change the program and run it again until nothing is listed under "Must fix
before indexing," the image files can be read, and "What the site does with
each column" reads every column as you meant. A column of names, places, or
dates that the site reads as notes needs a name it knows.

Then let the program read everything, in the background if it takes more than
a few minutes, and continue with step 2 of SKILL.md, with
`data/sources/NAME/manifest.parquet` as the source.

The terms decide two things the profile cannot know. When they forbid copies of
the images, index with `--images link`, whatever the profile recommends. When
they ask for a slow pace, give `ingest` a low `--download-workers`, such as 1,
since it fetches 16 images at once by default.

## Items elsewhere and vectors

- If the records link to the same objects elsewhere, such as photographs on
  Flickr, write them to `related.parquet` beside the manifest, with the columns
  the website lists, and pass `--related data/sources/NAME/related.parquet` to
  the profile and to `ingest`.
- If the images already have vectors from the model in `config.json`, as when
  the collection was indexed before, give them in a `vector` column. Indexing
  checks 20 of them against the model and computes only the ones missing.

## Tell the person

Say where the program is, that running it again brings the collection up to
date, and that it can move to a repository of its own. Say what the terms ask:
the credit, which goes in `collection.json`, any limit on how the site may be
used, and how long the data may be kept, with the date by which to run the
program again or delete the collection.
