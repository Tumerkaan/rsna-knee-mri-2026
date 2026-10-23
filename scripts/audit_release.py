"""Conservative local release scan. Findings contain rule/file only, never secrets."""
from pathlib import Path
import csv
import json
import re
import ast
import hashlib
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote

ROOT=Path(__file__).resolve().parents[1]
GENERATED_IMAGES={"fold-cv.png","target-heatmap.png","lb-timeline.png","residual-spearman.png","pipeline.png"}
SCHEMAS={
 "fold-cv.csv":["family","fold","macro_auc","cases","target_count","evidence"],
 "target-family-auc.csv":["family","fold","target","auc","evidence"],
 "residual-spearman.csv":["pair","fold","target","spearman","evidence"],
 "lb-history.csv":["date","submission","model","public_auc","status","final_slot","private_auc"],
 "gates.csv":["experiment","metric","value","criterion","original_decision","outcome"]}
SKIP={"__pycache__",".pytest_cache",".git","build","dist"}
TEXT_EXT={".md",".py",".json",".csv",".toml",".html",".css",".js",".txt"}
PATTERNS={
 "private_host_path":r"(?i)(?:[a-z]:[/\\](?:users|home)[/\\]|/(?:Users|home|content/drive)/)",
 "drive_document_link":r"https://(?:colab\.research\.google\.com/drive|drive\.google\.com)/",
 "signed_query":r"(?i)(?:X-Amz-Signature|X-Goog-Signature|GoogleAccessId|Signature)=[A-Za-z0-9%+/]{12,}",
 "credential_value":r"(?i)(?:api[_-]?key|api[_-]?token|password|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-+/]{18,}['\"]",
 "credential_prefix":r"(?:AKIA[A-Z0-9]{16}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9]{20,})",
 "dicom_identity_value":r"(?<![\w.])(?:1\.2\.840|2\.25)\.[0-9.]{12,}",
 "embedded_payload":r"[A-Za-z0-9+/]{600,}={0,2}",
 "private_key":r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"}

def candidate_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and not any(v in SKIP or v.endswith(".egg-info") for v in p.relative_to(ROOT).parts))

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[];self.images=[];self.scripts=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if "id" in a:self.ids.append(a["id"])
        if tag in ("a","link","script","img","iframe"):
            value=a.get("href") or a.get("src")
            if value:self.links.append(value)
        if tag=="img":self.images.append(a)
        if tag=="script":self.scripts.append(a.get("src",""))

def audit():
    findings=[];files=candidate_files()
    def fail(p,rule):findings.append({"file":p.relative_to(ROOT).as_posix(),"rule":rule})
    media_path=ROOT/"MEDIA_MANIFEST.json"
    media={}
    try:
        manifest=json.loads(media_path.read_text(encoding="utf-8"))
        assert manifest["schema"]=="ILLUSTRATION_ALLOWLIST_V1"
        for entry in manifest["files"]:
            path=entry["path"]
            assert path.startswith("docs/assets/") and path.endswith(".png")
            assert ".." not in Path(path).parts and path not in media
            assert re.fullmatch(r"[a-f0-9]{64}",entry["sha256"])
            assert entry["source"].startswith("https://")
            assert entry["license"] in {"CC BY 4.0","CC BY-SA 4.0","CC BY-NC-ND 4.0"}
            assert entry["author"] and entry["attribution"]=="ATTRIBUTION.md"
            assert (ROOT/path).is_file()
            media[path]=entry
    except (OSError,ValueError,KeyError,TypeError,AssertionError):
        fail(media_path,"illustration_manifest_invalid")
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix==".png":
            if rel in media:
                if hashlib.sha256(p.read_bytes()).hexdigest()!=media[rel]["sha256"]:
                    fail(p,"illustration_hash_mismatch")
            elif p.parent!=ROOT/"docs/assets" or p.name not in GENERATED_IMAGES:
                fail(p,"unapproved_image")
            continue
        if p.suffix not in TEXT_EXT and p.name not in ("LICENSE","NOTICE",".gitignore",".nojekyll"):fail(p,"unexpected_file_type");continue
        text=p.read_text(encoding="utf-8")
        for name,pattern in PATTERNS.items():
            if re.search(pattern,text):fail(p,name)
        if p.suffix==".py":
            try:ast.parse(text)
            except SyntaxError:fail(p,"python_syntax")
        if p.suffix==".json":
            try:json.loads(text,parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
            except (ValueError,TypeError):fail(p,"json_schema")
        if p.suffix==".csv":
            r=list(csv.reader(text.splitlines()))
            if p.parent!=ROOT/"results" or p.name not in SCHEMAS or not r or r[0]!=SCHEMAS[p.name]:fail(p,"non_aggregate_csv")
            elif len(r)>400 or any(len(row)!=len(r[0]) for row in r):fail(p,"csv_row_contract")
        if p.suffix==".html":
            h=Links();h.feed(text)
            if len(h.ids)!=len(set(h.ids)):fail(p,"duplicate_html_id")
            for link in h.links:
                u=urlsplit(link)
                if u.scheme:continue
                # This retained template is rendered into docs/index.html; its
                # links intentionally resolve relative to the output directory.
                base=ROOT/"docs" if rel=="docs/_build/index.template.html" else p.parent
                target=(base/unquote(u.path)).resolve() if u.path else p
                if not target.is_relative_to(ROOT) or not target.exists():fail(p,"broken_local_link")
                if not u.path and u.fragment and u.fragment not in h.ids:fail(p,"missing_anchor")
            # An empty, source-less lightbox placeholder receives both src and
            # alt from the selected image. Visible sourced images still need alt.
            if any("alt" not in i or (i.get("src") and not i["alt"]) for i in h.images):fail(p,"missing_image_alt")
            if any(urlsplit(s).scheme for s in h.scripts):fail(p,"external_script")
    # Explicit absence checks concern artifacts, not necessary disclosure words.
    required=["README.md","LICENSE","ATTRIBUTION.md","REPRODUCE.md","docs/WRITEUP.md",
              "docs/index.html","docs/KAGGLE_DISCUSSION.md","src/rsna_ensemble/inference.py"]
    for name in required:
        if not (ROOT/name).exists():findings.append({"file":name,"rule":"missing_required"})
    for name in GENERATED_IMAGES:
        if not (ROOT/"docs/assets"/name).is_file():findings.append({"file":name,"rule":"missing_figure"})
    return {"status":"PASS" if not findings else "FAIL","files_scanned":len(files),
            "aggregate_csv_files":len(list((ROOT/"results").glob("*.csv"))),
            "generated_figures":len(list((ROOT/"docs/assets").glob("*.png"))),
            "licensed_illustrations":len(media),
            "findings":findings,"published":False,
            "boundary":"Static allowlist and pattern scan, not a guarantee against every possible disclosure"}

if __name__=="__main__":
    result=audit();print(json.dumps(result,indent=2))
    raise SystemExit(result["status"]!="PASS")
