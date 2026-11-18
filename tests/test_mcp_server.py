import base64
import json
from pathlib import Path

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from PIL import Image
from starlette.applications import Starlette

from tests.test_catalog_filter import add_catalog_values
from tests.test_grouped_collection import (  # noqa: F401
    FLICKR_PAGE,
    PAGE,
    SaturatingModel,
    add_record_details,
    add_related_sources,
    grouped,
    picture,
    site,
)

pytestmark = pytest.mark.usefixtures("site")


def call(tool, arguments=None):
    from src.backend.api import mcp_server

    async def run():
        async with Client(mcp_server.server) as client:
            return await client.call_tool(tool, arguments or {})

    return anyio.run(run)


def call_over_http(tool, arguments, address="http://localhost:8000", headers=None):
    from src.backend.api import mcp_server

    app = Starlette(routes=mcp_server.http_routes())

    async def run():
        async with mcp_server.server.session_manager.run():
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(
                transport=transport, base_url=address, headers=headers
            ) as http:
                connection = streamable_http_client(f"{address}/mcp", http_client=http)
                async with Client(connection) as client:
                    return await client.call_tool(tool, arguments)

    return anyio.run(run)


def post_to_mcp(address="http://localhost:8000", headers=None, content=None):
    from src.backend.api import mcp_server

    app = Starlette(routes=mcp_server.http_routes())
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}

    async def run():
        async with mcp_server.server.session_manager.run():
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(
                transport=transport, base_url=address
            ) as http:
                return await http.post(
                    "/mcp",
                    content=content or json.dumps(ping).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        **(headers or {}),
                    },
                )

    return anyio.run(run)


def configure(folder, **values):
    (folder / "settings.json").write_text(json.dumps(values))


def first_page(answer):
    return answer.structured_content["results"][0]["page"]


QUILTS = {"query": "quilt", "groups": ["west"]}


def test_every_tool_only_reads():
    from src.backend.api import mcp_server

    async def run():
        async with Client(mcp_server.server) as client:
            return (await client.list_tools()).tools

    tools = anyio.run(run)
    assert [t.name for t in tools] == [
        "describe_collection",
        "search_collection",
        "find_similar",
        "search_by_image",
        "list_catalog_values",
        "get_object",
    ]
    assert all(t.annotations.read_only_hint for t in tools)
    assert all(t.output_schema for t in tools)


def test_lists_of_results_come_with_an_image_wall_for_hosts_that_show_apps():
    from src.backend.api import mcp_server

    async def run():
        async with Client(mcp_server.server) as client:
            tools = (await client.list_tools()).tools
            resources = (await client.list_resources()).resources
            page = await client.read_resource("ui://dce/wall.html")
            return tools, resources, page

    tools, resources, page = anyio.run(run)
    walls = {t.name for t in tools if (t.meta or {}).get("ui")}
    assert walls == {"search_collection", "find_similar", "search_by_image"}
    assert {t.meta["ui"]["resourceUri"] for t in tools if t.name in walls} == {
        "ui://dce/wall.html"
    }
    assert [(r.uri, r.mime_type) for r in resources] == [
        ("ui://dce/wall.html", "text/html;profile=mcp-app")
    ]
    assert "ui/initialize" in page.contents[0].text
    assert "never pair a title" in mcp_server.INSTRUCTIONS


def test_the_wall_may_load_pictures_only_from_the_sites_own_address(monkeypatch):
    from src.backend.api import mcp_server
    from src.backend.core.config import settings

    def allowed():
        meta = mcp_server.image_wall().resources()[0].resource.meta or {}
        return meta.get("ui", {}).get("csp", {}).get("resourceDomains")

    monkeypatch.setattr(settings, "public_url", "https://collection.example.org/maps")
    assert allowed() == ["https://collection.example.org"]
    monkeypatch.setattr(settings, "public_url", "")
    assert allowed() is None


def test_the_wall_may_also_load_pictures_from_the_servers_that_keep_them(
    monkeypatch,
):
    from src.backend.api import mcp_server
    from src.backend.core.config import settings

    def allowed():
        (binding,) = mcp_server.wall_app.resources()
        return ((binding.resource.meta or {}).get("ui") or {}).get("csp")

    monkeypatch.setattr(settings, "public_url", "https://collection.example.org/maps")
    try:
        mcp_server.allow_images_from(["https://tile.loc.gov"])
        assert allowed() == {
            "resourceDomains": [
                "https://collection.example.org",
                "https://tile.loc.gov",
            ]
        }
        monkeypatch.setattr(settings, "public_url", "")
        mcp_server.allow_images_from([])
        assert allowed() is None
    finally:
        mcp_server.allow_images_from([])


def test_a_result_kept_elsewhere_shows_its_thumbnail_from_there():
    from src.backend.api import mcp_server

    address = "https://tile.loc.gov/iiif/a/full/!400,400/0/default.jpg"
    result = {"id": "a", "metadata": {"image_urls": {"thumbnail": address}}}
    assert mcp_server.found(result, {}, "https://site.org").thumbnail == address
    plain = mcp_server.found({"id": "b", "metadata": {}}, {}, "https://site.org")
    assert plain.thumbnail == "https://site.org/images/b?size=thumbnail"


def test_the_collection_is_described_with_its_groups():
    summary = call("describe_collection").structured_content
    assert [(g["id"], g["label"]) for g in summary["groups"]] == [
        ("east", "East Wing"),
        ("west", "West Wing"),
    ]
    assert summary["images_without_record"] == 1


def test_a_search_names_each_object_and_links_to_its_page():
    answer = call("search_collection", {"query": "quilt", "groups": ["west"]})
    results = answer.structured_content["results"]
    assert [r["id"] for r in results] == ["b1", "b2"]
    assert results[0]["group"] == "West Wing"
    assert results[0]["page"] == "/#/item/b1"
    assert results[0]["thumbnail"] == "/images/b1?size=thumbnail"


def test_results_name_their_object_and_later_pages_can_be_reached():
    first = call(
        "search_collection", {"query": "quilt", "groups": ["west"], "limit": 1}
    )
    second = call(
        "search_collection",
        {"query": "quilt", "groups": ["west"], "limit": 1, "page": 2},
    )
    assert [r["id"] for r in first.structured_content["results"]] == ["b1"]
    assert [r["id"] for r in second.structured_content["results"]] == ["b2"]
    assert first.structured_content["results"][0]["object_id"] == "rec-b"


def test_a_search_can_leave_out_images_without_a_record():
    answer = call(
        "search_collection",
        {"query": "quilt", "groups": ["west"], "cataloged_only": True},
    )
    assert [r["id"] for r in answer.structured_content["results"]] == ["b1"]


def test_mistakes_come_back_as_errors_the_model_can_read():
    unknown_group = call("search_collection", {"query": "quilt", "groups": ["nowhere"]})
    assert unknown_group.is_error and "nowhere" in unknown_group.content[0].text
    unknown_object = call("get_object", {"id": "nope"})
    assert unknown_object.is_error and "nope" in unknown_object.content[0].text
    too_many = call("search_collection", {"query": "quilt", "limit": 500})
    assert too_many.is_error


def test_an_object_carries_its_record_and_where_else_it_appears():
    from src.backend.api.routes import browse

    embeddings_dir = browse.collection_service.embeddings.embeddings_dir
    add_record_details(embeddings_dir)
    add_related_sources(embeddings_dir.parent)
    details = call("get_object", {"id": "a1"}).structured_content
    assert details["record"] == PAGE
    assert details["fields"][0] == {"label": "Credit Line", "value": "Gift of a donor"}
    assert details["elsewhere"][0]["url"] == FLICKR_PAGE
    assert details["images"][0] == "/images/a1?size=full"


def test_similar_objects_say_why_and_how_alike_they_are():
    similar = call("find_similar", {"id": "a1", "limit": 2}).structured_content
    assert len(similar["results"]) == 2
    assert {r["why"] for r in similar["results"]} == {"Looks similar"}
    assert all(isinstance(r["similarity"], float) for r in similar["results"])


def test_similar_objects_can_come_from_some_groups_only():
    similar = call("find_similar", {"id": "a1", "groups": ["west"]}).structured_content
    assert [r["id"] for r in similar["results"]] == ["b1", "b2"]


def with_catalog_values():
    from src.backend.api.routes import browse

    add_catalog_values(browse.collection_service.embeddings.embeddings_dir)


def test_searches_can_keep_objects_with_some_catalog_values():
    assert not call("describe_collection").structured_content["catalog_filters"]
    assert call(
        "search_collection", {"query": "quilt", "catalog": {"place": ["Ohio"]}}
    ).is_error
    with_catalog_values()
    assert call("describe_collection").structured_content["catalog_filters"]
    ohio = call("search_collection", {"query": "quilt", "catalog": {"place": ["Ohio"]}})
    assert [r["object_id"] for r in ohio.structured_content["results"]] == ["rec-a"]
    early = call(
        "find_similar", {"id": "a1", "catalog": {"year_from": 1850, "year_to": 1869}}
    )
    assert [r["id"] for r in early.structured_content["results"]] == ["b1"]
    backwards = call(
        "search_collection",
        {"query": "quilt", "catalog": {"year_from": 1900, "year_to": 1800}},
    )
    assert backwards.is_error and "year_from" in backwards.content[0].text


def test_the_catalog_values_to_filter_by_can_be_listed():
    with_catalog_values()
    listed = call("list_catalog_values", {"category": "topic", "contains": "quilt"})
    assert listed.structured_content == {
        "category": "topic",
        "values": [{"value": "Quilts", "records": 1}],
    }
    assert call("list_catalog_values", {"category": "color"}).is_error


def test_the_instructions_explain_query_fields_and_catalog_filters():
    from src.backend.api import mcp_server

    assert "date:1860s" in mcp_server.INSTRUCTIONS
    assert "list_catalog_values" in mcp_server.INSTRUCTIONS


@pytest.fixture()
def stored_images(monkeypatch):
    from src.backend.api.routes import search
    from src.backend.core.config import settings

    monkeypatch.setattr(search, "model_service", SaturatingModel())
    folder = Path(settings.data_dir) / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48)).save(folder / "a1.jpg")


def test_a_collection_image_or_part_of_it_can_be_the_query(stored_images):
    whole = call("search_by_image", {"id": "a1"}).structured_content["results"]
    assert [r["id"] for r in whole] == ["a2", "b1", "a3", "b2"]
    face = {"left": 0.25, "top": 0.25, "width": 0.5, "height": 0.5}
    part = call(
        "search_by_image",
        {"id": "a1", "crop": face, "groups": ["west"], "cataloged_only": True},
    )
    assert [r["id"] for r in part.structured_content["results"]] == ["b1"]
    too_small = {"left": 0.0, "top": 0.0, "width": 0.1, "height": 0.1}
    assert call("search_by_image", {"id": "a1", "crop": too_small}).is_error
    with_catalog_values()
    english = call("search_by_image", {"id": "a1", "catalog": {"place": ["England"]}})
    assert [r["id"] for r in english.structured_content["results"]] == ["a3"]


def test_an_image_of_the_assistants_own_can_be_the_query(stored_images):
    sent = base64.b64encode(picture()).decode()
    found = call("search_by_image", {"image_data": sent}).structured_content
    assert [r["id"] for r in found["results"]] == ["a2", "b1", "a3", "b2"]
    assert call("search_by_image", {"image_data": "not an image"}).is_error
    assert call("search_by_image", {}).is_error
    assert call("search_by_image", {"id": "a1", "image_data": sent}).is_error
    assert call("search_by_image", {"id": "nope"}).is_error


def test_the_endpoint_answers_over_http_with_links_to_the_public_address(monkeypatch):
    from src.backend.core.config import settings

    seen_here = call_over_http("search_collection", QUILTS)
    assert first_page(seen_here) == "http://localhost:8000/#/item/b1"
    monkeypatch.setattr(settings, "public_url", "https://collection.example.org")
    seen_there = call_over_http("search_collection", QUILTS)
    assert first_page(seen_there) == "https://collection.example.org/#/item/b1"


def test_links_take_https_from_the_proxy_only_when_one_is_declared(tmp_path):
    https = {"X-Forwarded-Proto": "https"}
    assert first_page(call_over_http("search_collection", QUILTS, headers=https)) == (
        "http://localhost:8000/#/item/b1"
    )
    configure(tmp_path, proxy_hops=1)
    assert first_page(call_over_http("search_collection", QUILTS, headers=https)) == (
        "https://localhost:8000/#/item/b1"
    )
    renamed = {**https, "X-Forwarded-Host": "collection.example.org"}
    similar = {"id": "a1", "groups": ["west"]}
    assert first_page(call_over_http("find_similar", similar, headers=renamed)) == (
        "https://collection.example.org/#/item/b1"
    )
    odd = {"X-Forwarded-Proto": "javascript"}
    assert first_page(call_over_http("search_collection", QUILTS, headers=odd)) == (
        "http://localhost:8000/#/item/b1"
    )


def test_the_public_address_wins_over_what_the_proxy_says(tmp_path, monkeypatch):
    from src.backend.core.config import settings

    configure(tmp_path, proxy_hops=1)
    monkeypatch.setattr(settings, "public_url", "https://library.example.org/maps")
    forwarded = {"X-Forwarded-Proto": "http", "X-Forwarded-Host": "elsewhere.example"}
    answer = call_over_http("search_collection", QUILTS, headers=forwarded)
    assert first_page(answer) == "https://library.example.org/maps/#/item/b1"


def test_assistants_have_a_search_limit_of_their_own(site, tmp_path):
    configure(tmp_path, searches_per_minute=1, mcp_searches_per_minute=2, proxy_hops=1)
    one_address = {"X-Forwarded-For": "198.51.100.7"}
    assert not call_over_http("search_collection", QUILTS, headers=one_address).is_error
    similar = call_over_http("find_similar", {"id": "a1"}, headers=one_address)
    assert not similar.is_error
    third = call_over_http("search_collection", QUILTS, headers=one_address)
    assert third.is_error and "wait a moment" in third.content[0].text
    visitor = [site.get("/api/search/text?query=quilt", headers=one_address)]
    visitor.append(site.get("/api/search/text?query=quilt", headers=one_address))
    assert [answer.status_code for answer in visitor] == [200, 429]
    configure(tmp_path, mcp_searches_per_minute=0, proxy_hops=1)
    assert not call_over_http("search_collection", QUILTS, headers=one_address).is_error


def test_the_endpoint_refuses_requests_over_four_megabytes():
    answer = post_to_mcp(content=b"x" * (4 * 1024 * 1024 + 1))
    assert answer.status_code == 413


def test_a_page_on_another_site_is_turned_away():
    for origin in ("https://elsewhere.example", "http://localhost.elsewhere.example"):
        answer = post_to_mcp(headers={"Origin": origin})
        assert answer.status_code == 403
        assert "other sites" in answer.text


def test_assistants_which_send_no_origin_are_answered():
    assert post_to_mcp().json() == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_an_origin_of_null_is_turned_away():
    assert post_to_mcp(headers={"Origin": "null"}).status_code == 403


def test_pages_on_this_computer_and_on_the_site_itself_may_call_it(monkeypatch):
    from src.backend.core.config import settings

    for origin in ("http://localhost:5173", "https://127.0.0.1:3000", "http://[::1]"):
        assert post_to_mcp(headers={"Origin": origin}).status_code == 200
    by_ip = "http://203.0.113.10:8000"
    assert post_to_mcp(by_ip, headers={"Origin": by_ip}).status_code == 200
    other_scheme = {"Origin": "https://203.0.113.10:8000"}
    assert post_to_mcp(by_ip, headers=other_scheme).status_code == 403
    monkeypatch.setattr(settings, "public_url", "https://library.example.org/maps")
    public = {"Origin": "https://library.example.org"}
    assert post_to_mcp(by_ip, headers=public).status_code == 200


def test_a_name_other_than_the_sites_own_is_turned_away():
    answer = post_to_mcp("http://rebound.example:8000")
    assert answer.status_code == 421
    assert "Set DCE_PUBLIC_URL to the address visitors use" in answer.text


def test_the_name_in_the_public_address_is_answered_at_any_port(monkeypatch):
    from src.backend.core.config import settings

    monkeypatch.setattr(settings, "public_url", "https://collection.example.org/maps")
    for address in (
        "http://collection.example.org",
        "http://collection.example.org:81",
    ):
        assert post_to_mcp(address).status_code == 200
    assert post_to_mcp("http://other.example.org").status_code == 421


@pytest.mark.parametrize(
    "address",
    [
        "http://203.0.113.10",
        "http://203.0.113.10:8000",
        "http://[2001:db8::1]:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
    ],
)
def test_an_ip_address_or_localhost_is_answered(address):
    assert post_to_mcp(address).status_code == 200


def test_any_name_is_answered_behind_a_declared_proxy(tmp_path):
    configure(tmp_path, proxy_hops=1)
    machine = "http://ec2-203-0-113-10.us-west-2.compute.amazonaws.com"
    assert post_to_mcp(machine).status_code == 200
    forwarded = {"X-Forwarded-Proto": "https", "X-Forwarded-Host": "c.example.org"}
    own = {**forwarded, "Origin": "https://c.example.org"}
    assert post_to_mcp(machine, headers=own).status_code == 200
    elsewhere = {**forwarded, "Origin": "https://elsewhere.example"}
    assert post_to_mcp(machine, headers=elsewhere).status_code == 403


def test_the_servers_that_keep_an_image_are_known():
    from src.backend.services.catalog_fields import image_hosts

    assert image_hosts(
        {
            "paths": {"thumbnail": "thumbnails/a.jpg"},
            "iiif": "https://tile.loc.gov/iiif/a/info.json",
        }
    ) == {"https://tile.loc.gov"}
    assert image_hosts({"paths": {"thumbnail": "thumbnails/a.jpg"}}) == set()


def test_an_object_names_its_groups_institution_and_license(tmp_path):
    cc0 = {"name": "CC0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}
    (tmp_path / "collection.json").write_text(
        json.dumps(
            {
                "license": {"name": "CC BY 4.0"},
                "groups": {
                    "west": {
                        "label": "West Museum",
                        "institution": "the West Museum",
                        "license": cc0,
                    }
                },
            }
        )
    )
    summary = call("describe_collection").structured_content
    west = next(g for g in summary["groups"] if g["id"] == "west")
    assert (west["label"], west["institution"]) == ("West Museum", "the West Museum")
    assert west["license"] == f"CC0 {cc0['url']}"
    rifle = call("get_object", {"id": "b1"}).structured_content
    assert (rifle["group"], rifle["institution"]) == ("West Museum", "the West Museum")
    assert rifle["license"] == west["license"]
    quilt = call("get_object", {"id": "a1"}).structured_content
    assert (quilt["institution"], quilt["license"]) == (None, "CC BY 4.0")
