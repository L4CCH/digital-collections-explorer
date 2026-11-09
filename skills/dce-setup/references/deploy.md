# Putting the site on a server or AWS

Only when the person asks. The full guides are at
https://l4cch.github.io/dce-website/docs/deploy/server/ and
https://l4cch.github.io/dce-website/docs/deploy/aws/.

Both index with copies of every image. For `thumbnails` or `link`, index on this
machine as in step 5 and start the site from that folder with `--from`.

## Any Linux machine

```bash
sudo deploy/bootstrap.sh install --name NAME --source /path/to/images
sudo deploy/bootstrap.sh install --name NAME --from data/collections/NAME
sudo deploy/bootstrap.sh describe --name NAME --collection-file collection.json
sudo deploy/bootstrap.sh status --name NAME
```

The first indexes the images on the machine, the second serves a folder indexed
elsewhere. Either keeps the site running across reboots.

## AWS

```bash
deploy/aws/deploy.sh NAME --source s3://bucket/prefix --anonymous
```

The images must already be reachable from the cloud, at an `s3://` or
`https://` address. `--from s3://bucket/folder` starts from a folder indexed
elsewhere instead.

`deploy.sh` prints what it will create and what that costs per month, then asks
for `yes`. Your shell cannot answer, so the first run stops there having changed
nothing. Show the person that list. Only after they say yes to it, run the same
command again with `--yes`.

- **Who can open it.** Without `--https` the site is open to everyone at the
  machine's address over plain HTTP. To keep it internal, pass `--allowed-cidr`
  with the institution's address range. `--https` serves it to the public through
  CloudFront, and cannot be combined with `--allowed-cidr`. `--domain` with
  `--certificate-arn` also serves it at a domain of their own.
- Index on a large machine and serve from a small one: run `deploy.sh` again
  with a smaller `--instance-type`. It keeps the index and the address.
- `status.sh NAME` reports progress and cost. `describe.sh NAME --file
  collection.json` changes the title, introduction and suggested searches;
  `configure.sh` changes the limits; `update.sh` switches versions; `destroy.sh`
  removes everything it created, after the same confirmation as `deploy.sh`.
- `mcp.sh NAME --on` lets AI assistants search the site at `/mcp`. Assistants
  that call it from the cloud, such as Claude and ChatGPT connectors, need a
  public `https://` address, so deploy with `--https`.
