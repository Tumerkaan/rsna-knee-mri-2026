"""Synthetic contract tests. No real images, labels, weights, training or network."""
import importlib
import json
import time
import types
import dis
import builtins
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
import pytest
import torch

from rsna_ensemble import robust
from rsna_ensemble.inference import own_fallback, atomic_predictions, load_model
from rsna_ensemble.local_runtime import save_failure, read_bundle

FAMILIES=("abig","t1","f","g","z")

@pytest.mark.parametrize("family",FAMILIES)
def test_training_import_has_no_unbound_globals(family):
    m=importlib.import_module(f"rsna_ensemble.families.{family}.training")
    bound=set("OUT START STARTED DEADLINE WORK_DEADLINE WALL_SECONDS FOLD DATA CONFIG TARGETS SEEDS WEIGHT_PATH WEIGHT_SHA check_deadline _check fold_rows sha_file _sha _save _copy_milestone CANDIDATE DTYPE PRECISION".split())
    def codes(c):
        yield c
        for v in c.co_consts:
            if isinstance(v,types.CodeType):
                yield from codes(v)
    functions=[v for v in vars(m).values() if isinstance(v,types.FunctionType) and v.__module__==m.__name__]
    assert functions
    for fn in functions:
        for c in codes(fn.__code__):
            assert not {i.argval for i in dis.get_instructions(c) if i.opname=="LOAD_GLOBAL"
                        and i.argval not in vars(m) and i.argval not in vars(builtins) and i.argval not in bound}

@pytest.mark.parametrize("family",FAMILIES)
def test_slot_triplets_do_not_cross_boundaries(family):
    geo=importlib.import_module(f"rsna_ensemble.families.{family}.geometry")
    starts,planes=geo.valid_windows(np.ones(96,dtype=np.uint8))
    assert len(starts)==86 and len(planes)==86
    bounds=np.array([0,26,47,65,78,96])
    for s in starts:
        slot=np.searchsorted(bounds,s,side="right")-1
        assert s+2<bounds[slot+1]
    mask=np.ones(96,dtype=np.uint8);mask[10]=0
    starts,_=geo.valid_windows(mask)
    assert all(10 not in range(s,s+3) for s in starts)

@pytest.mark.parametrize("family",("abig","t1","f","z"))
def test_anatomical_mirror_involution_and_axis(family):
    mod=importlib.import_module(f"rsna_ensemble.families.{family}."+("model" if family=="f" else "recipe"))
    x=torch.arange(3*3*4*5,dtype=torch.float32).reshape(3,3,4,5)
    planes=torch.tensor([0,1,2])
    y=mod.mirror_windows(x,planes)
    assert torch.equal(y[0],x[0].flip(0))
    assert torch.equal(y[1],x[1].flip(-1))
    assert torch.equal(mod.mirror_windows(y,planes),x)

def test_metric_ties_silence_and_undefined():
    from rsna_ensemble.families.abig.recipe import cv_metrics
    y=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    p=np.array([[.1,.4],[.8,.4],[.8,.4],[.1,.4]])
    silent=np.zeros_like(y,dtype=bool)
    r=cv_metrics(y,p,silent,["fixture-a","fixture-b"])
    assert r["macro"]==.75 and r["denominator"]==2
    silent[:,1]=True
    r=cv_metrics(y,p,silent,["fixture-a","fixture-b"])
    assert r["denominator"]==1 and r["targets"][1]["auc"] is None

@pytest.mark.parametrize("family",FAMILIES)
def test_epoch1_saves_before_speed_stop(family):
    mod=importlib.import_module(f"rsna_ensemble.families.{family}."+("t1_policy" if family=="t1" else "recipe"))
    calls=[]
    def snapshot(epoch):
        calls.append(epoch)
        return {"validation_seconds":2.,"write_seconds":1.}
    r=mod.finish_epoch(1,0,25,snapshot,{"validation_seconds":0.,"write_seconds":0.},clock=lambda:10.)
    assert calls==[1] and r["stop"] and r["snapshot_saved"] and not r["scheduler_reset"]

def test_ema_skipped_step_never_updates():
    from rsna_ensemble.families.t1.t1_policy import ModelEMA
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1)
    ema=ModelEMA(model,decay=.5);model.weight.data.fill_(3)
    ema.update(model,successful=False)
    assert ema.updates==0 and ema.module.weight.item()==1
    ema.update(model,successful=True)
    assert ema.updates==1 and ema.module.weight.item()==2

def test_oom_bounded_and_single_precision_recovery():
    attempts=[]
    def call(batch):
        attempts.append(batch)
        if batch>2:raise torch.cuda.OutOfMemoryError("fixture")
        return batch
    assert robust.retry_oom(call,initial=16,release=lambda:None,deadline=time.monotonic()+10)==2
    assert attempts==[16,8,4,2]
    counters={"fp32_retries":0,"nonfinite_f_skips":0}
    def precision(fp32):
        if not fp32:raise robust.NonfiniteLogits("fixture")
        return 1
    assert robust.retry_precision(precision,counters)==1
    assert counters["fp32_retries"]==1 and counters["fp32_recoveries"]==1

def test_unrecoverable_numeric_stops_after_one_fp32_attempt():
    calls=[];counters={"fp32_retries":0,"nonfinite_f_skips":0}
    def bad(fp32):
        calls.append(fp32);raise robust.NonfiniteLogits("fixture")
    with pytest.raises(robust.NonfiniteLogits):robust.retry_precision(bad,counters)
    assert calls==[False,True] and counters["nonfinite_f_skips"]==1

def test_fallback_modes_and_expired_deadline():
    prior=np.full(12,.3)
    def good():return np.full(12,.6)
    def bad():return np.full(12,np.nan)
    future=time.monotonic()+10
    assert own_fallback(good,good,good,prior,deadline=future)[1]=="C0_FULL"
    assert own_fallback(good,good,bad,prior,deadline=future)[1]=="ABT_FALLBACK"
    assert own_fallback(bad,good,good,prior,deadline=future)[1]=="TRAINING_PREVALENCE"
    assert own_fallback(good,good,good,prior,deadline=0)[1]=="TRAINING_PREVALENCE"

def test_atomic_writer_rejects_bad_output_without_replacement(tmp_path):
    path=tmp_path/"out.csv"
    ids=["fixture-a","fixture-b"]
    atomic_predictions(path,ids,np.full((2,12),.5))
    before=path.read_bytes()
    for values,keys in [(np.full((2,12),np.nan),ids),(np.full((2,12),.5),["fixture-a"]*2)]:
        with pytest.raises(ValueError):atomic_predictions(path,keys,values)
        assert path.read_bytes()==before
    assert not path.with_suffix(".partial").exists()

def test_bundle_bad_hash_stops_before_array_load(tmp_path,monkeypatch):
    path=tmp_path/"artifact";path.write_text("synthetic")
    names=("volumes","masks","labels","silent","folds","ids","initial_weights")
    manifest=tmp_path/"bundle.json"
    manifest.write_text(json.dumps({n:{"path":path.name,"sha256":"invalid"} for n in names}))
    monkeypatch.setattr(np,"load",lambda *a,**k:pytest.fail("array opened before hash validation"))
    with pytest.raises(ValueError,match="bundle_hash"):read_bundle(manifest,384,time.monotonic()+10)

def test_checkpoint_bad_hash_stops_before_deserialization(tmp_path,monkeypatch):
    path=tmp_path/"checkpoint";path.write_text("fixture")
    monkeypatch.setattr(torch,"load",lambda *a,**k:pytest.fail("deserialization before hash validation"))
    with pytest.raises(ValueError,match="checkpoint_hash"):load_model("abig",path,"invalid",0)

def test_source_error_redacts_values_and_keeps_location(tmp_path):
    secret="synthetic"+"-credential-value"
    try:raise ValueError(secret)
    except ValueError as exc:r=save_failure(exc,"PUBLICATION_TEST",tmp_path)
    s=json.dumps(r)
    assert secret not in s and r["locals_recorded"] is False and r["traceback_available"]
    assert r["exception_type"]=="ValueError" and r["traceback"][-1]["line"]>0

def test_cli_defaults_to_no_training():
    env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src")}
    result=subprocess.run([sys.executable,"-m","rsna_ensemble.train","--family","abig","--fold","0",
        "--bundle","unused.json","--output","unused-output","--wall-seconds","72000"],capture_output=True,text=True,env=env)
    assert result.returncode==0 and "NOT_RUN" in result.stdout

def test_child_source_failure_preserved_by_supervisor(tmp_path):
    env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src")}
    out=tmp_path/"new-run"
    result=subprocess.run([sys.executable,"-m","rsna_ensemble.train","--family","abig","--fold","0",
        "--bundle",str(tmp_path/"missing.json"),"--output",str(out),"--wall-seconds","600","--execute"],
        capture_output=True,text=True,env=env,timeout=30)
    assert result.returncode==1
    source=json.loads((out/"source-error.json").read_text())
    parent=json.loads((out/"supervisor.json").read_text())
    assert source["exception_type"]=="FileNotFoundError"
    assert source["traceback_available"] and source["locals_recorded"] is False
    assert "source-error.json" in parent["source_error_records"]
    assert parent["automatic_retry"] is False and parent["exit_code"]==1
