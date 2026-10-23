"""Release boundary regression tests, using only tiny synthetic fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location("release_audit",Path(__file__).resolve().parents[1]/"scripts/audit_release.py")
audit=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

@pytest.fixture
def release(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,"ROOT",tmp_path)
    (tmp_path/"docs/assets").mkdir(parents=True)
    (tmp_path/"results").mkdir()
    (tmp_path/"MEDIA_MANIFEST.json").write_text(json.dumps({"schema":"ILLUSTRATION_ALLOWLIST_V1","files":[]}))
    return tmp_path

def rules():
    return {f["rule"] for f in audit.audit()["findings"]}

def test_arbitrary_image_is_rejected(release):
    (release/"docs/assets/unapproved.png").write_bytes(b"synthetic")
    assert "unapproved_image" in rules()

def test_listed_image_is_bound_to_exact_bytes(release):
    image=release/"docs/assets/fixture.png"
    image.write_bytes(b"synthetic")
    entry={"path":"docs/assets/fixture.png","sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
           "source":"https://example.org/fixture","license":"CC BY 4.0","author":"Fixture", "attribution":"ATTRIBUTION.md"}
    (release/"MEDIA_MANIFEST.json").write_text(json.dumps({"schema":"ILLUSTRATION_ALLOWLIST_V1","files":[entry]}))
    assert not {"illustration_manifest_invalid","illustration_hash_mismatch","unapproved_image"}&rules()
    image.write_bytes(b"changed")
    assert "illustration_hash_mismatch" in rules()

def test_missing_manifest_fails_closed(release):
    (release/"MEDIA_MANIFEST.json").unlink()
    assert "illustration_manifest_invalid" in rules()

def test_sourced_images_need_alt_but_empty_placeholder_is_allowed(release):
    page=release/"docs/index.html"
    page.write_text('<img alt="">')
    assert "missing_image_alt" not in rules()
    page.write_text('<img src="assets/missing.png" alt="">')
    assert {"missing_image_alt","broken_local_link"}<=rules()
    page.write_text('<img>')
    assert "missing_image_alt" in rules()

def test_template_resolves_output_links_and_still_rejects_missing_links(release):
    folder=release/"docs/_build";folder.mkdir()
    page=folder/"index.template.html"
    page.write_text('<a href="assets/">Assets</a>')
    assert "broken_local_link" not in rules()
    page.write_text('<a href="assets/missing.png">Missing</a>')
    assert "broken_local_link" in rules()
