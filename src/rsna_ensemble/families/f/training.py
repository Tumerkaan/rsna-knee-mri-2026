import math, random, json, time, gc
from pathlib import Path
import numpy as np
import torch
from .recipe import *
from rsna_ensemble.local_runtime import json_save, save_failure, diagnostic
def resolve_precision(requested='fp16', *, cuda_available, bf16_supported):
    if not cuda_available:
        raise ValueError('F_CUDA_REQUIRED')
    if requested not in ('fp16', 'bf16', 'auto'):
        raise ValueError('F_PRECISION_INVALID')
    selected = ('bf16' if bf16_supported else 'fp16') if requested == 'auto' else requested
    if selected == 'bf16' and not bf16_supported:
        raise ValueError('F_BF16_UNSUPPORTED')
    return dict(requested=requested, selected=selected, grad_scaler_enabled=selected=='fp16',
                selection_basis='EXPLICIT_OR_CUDA_CAPABILITY', device_name_used_for_selection=False)

def execute_oom_fallback(compute_gradients, restore_unstepped_state, batch_state, *, oom_type):
    """Only a pre-step gradient computation may be retried, once, at batch four."""
    retry = False
    try:
        return compute_gradients(batch_state['physical_batch'])
    except oom_type:
        if batch_state['physical_batch'] != 8 or batch_state['oom_fallback_used']:
            raise
        retry = True
    # Leave the exception handler before retrying so the failed graph's traceback
    # cannot retain CUDA tensors. No optimizer step has occurred in this function.
    if retry:
        restore_unstepped_state()
        batch_state.update(physical_batch=4, grad_accum=2, oom_fallback_used=True)
        return compute_gradients(4)

def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def _seed_worker(_worker):
    from rsna_ensemble.error_record import install_child_hook
    from rsna_ensemble.local_runtime import error_files, CODES
    install_child_hook(OUT/f'fold-{FOLD}',stage='F_CORPUS_GETITEM',
                       allowed_files=error_files(),allowed_codes=CODES)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed);np.random.seed(seed)

def _dataset(rows, training, folder):
    from .recipe import CorpusCases
    from rsna_ensemble.local_runtime import save_failure
    class SafeCases(CorpusCases):
        def __getitem__(self,index):
            try:
                return super().__getitem__(index)
            except BaseException as exc:
                save_failure(exc,'F_CORPUS_GETITEM',folder,worker=True)
                raise
    return SafeCases(DATA['volumes'],DATA['masks'],DATA['mapping'],rows,training=training)

def _model_optimizer():
    from safetensors.torch import load_file
    from .model import build_model
    model = build_model(CANDIDATE,state=load_file(str(WEIGHT_PATH),device='cpu'),
                        backbone_microbatch=16,bn_policy='frozen').cuda()
    optimizer = torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=2e-4),
        dict(params=list(model.projection.parameters())+list(model.mil.parameters()),lr=1e-3)],weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda',init_scale=1024,enabled=PRECISION['grad_scaler_enabled'])
    return model,optimizer,scaler

def _train_epoch(model,optimizer,scaler,rows,epoch,completed,total_steps,batch_state,folder):
    from .recipe import case_batches,lr_multiplier
    from .model import soft_bce,bn_state
    model.train()
    before = bn_state(model)
    train = _dataset(rows,True,folder)
    indices = np.random.default_rng(SEEDS[FOLD]+epoch).permutation(len(train)).tolist()
    loader = torch.utils.data.DataLoader(train,batch_sampler=case_batches(indices),num_workers=8,
        pin_memory=True,worker_init_fn=_seed_worker,generator=torch.Generator().manual_seed(SEEDS[FOLD]+epoch))
    iterator = iter(loader)
    seen,losses,skipped,consecutive = 0,[],0,0
    began = time.monotonic()
    try:
        for x,valid,planes,batch_rows in iterator:
            _check()
            targets = torch.from_numpy(DATA['y'][batch_rows.numpy()].astype(np.float32))
            factor = lr_multiplier(completed,total_steps)
            for group,base in zip(optimizer.param_groups,(2e-4,1e-3)):
                group['lr'] = base*factor
            cpu_rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state()
            def compute(physical_batch):
                optimizer.zero_grad(set_to_none=True)
                observed = torch.isfinite(targets).sum().item()
                step_loss = 0.
                # Retain A's final-short-batch merge (8..15 cases) without
                # splitting it unless the explicitly authorized OOM fallback ran.
                chunk_size=len(x) if physical_batch==8 else 4
                chunks = [slice(i,min(i+chunk_size,len(x))) for i in range(0,len(x),chunk_size)]
                for sl in chunks:
                    bx,bv,by = x[sl].cuda(non_blocking=True),valid[sl].cuda(non_blocking=True),targets[sl].cuda()
                    with torch.autocast('cuda',dtype=DTYPE):
                        logits = model(bx,bv,torch.zeros(x.shape[1],dtype=torch.long,device=bx.device))
                        loss = soft_bce(logits.float(),by)*torch.isfinite(by).sum()/max(1,observed)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise ValueError('nonfinite_loss')
                    scaler.scale(loss).backward()
                    step_loss += float(loss.detach())
                    del bx,bv,by,logits,loss
                return step_loss
            def restore():
                optimizer.zero_grad(set_to_none=True)
                gc.collect();torch.cuda.empty_cache()
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
            loss_value = execute_oom_fallback(compute,restore,batch_state,oom_type=torch.cuda.OutOfMemoryError)
            scaler.unscale_(optimizer)
            finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            if not finite and not PRECISION['grad_scaler_enabled']:
                raise ValueError('nonfinite_gradient_stop')
            scaler.step(optimizer);scaler.update()
            completed += 1
            consecutive = 0 if finite else consecutive+1
            skipped += int(not finite)
            if consecutive > 8 or scaler.get_scale() < 1:
                raise ValueError('nonfinite_gradient_stop')
            losses.append(loss_value);seen += len(x)
            if time.monotonic()-began > 3600:
                raise TimeoutError('epoch_train_limit')
    finally:
        if hasattr(iterator,'_shutdown_workers'):
            iterator._shutdown_workers()
    after = bn_state(model)
    if seen != len(rows) or not all(torch.equal(before[key],after[key]) for key in before):
        raise ValueError('F_BN_OR_COHORT_DRIFT')
    return dict(completed_steps=completed,skip_count=skipped,mean_train_loss=float(np.mean(losses)),
                train_seconds=time.monotonic()-began,cases=seen,bn_running_statistics_unchanged=True)

def _predict(model,rows,folder,batch_state):
    from .model import mirror_windows
    dataset = _dataset(rows,False,folder)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_state['physical_batch'],shuffle=False,
        num_workers=4,pin_memory=True,worker_init_fn=_seed_worker)
    model.eval()
    outputs,order = [],[]
    iterator = iter(loader)
    try:
        with torch.inference_mode():
            for x,valid,planes,batch_rows in iterator:
                _check()
                # Same anatomical TTA/probability averaging as the sealed A.
                def predict(physical_batch):
                    chunks=[]
                    for i in range(0,len(x),physical_batch):
                        bx=x[i:i+physical_batch].cuda(non_blocking=True)
                        bv=valid[i:i+physical_batch].cuda(non_blocking=True)
                        bp=planes[i:i+physical_batch].cuda(non_blocking=True)
                        with torch.autocast('cuda',dtype=DTYPE):
                            first=model(bx,bv,torch.zeros(bx.shape[1],dtype=torch.long,device=bx.device))
                            mirrored=mirror_windows(bx.flatten(0,1),bp.flatten()).reshape_as(bx)
                            second=model(mirrored,bv,torch.zeros(bx.shape[1],dtype=torch.long,device=bx.device))
                            p=(first.float().sigmoid()+second.float().sigmoid())/2
                        if not torch.isfinite(p).all():
                            raise ValueError('nonfinite_prediction')
                        chunks.append(p.cpu().numpy())
                        del bx,bv,bp,first,mirrored,second,p
                    return np.concatenate(chunks)
                def restore():
                    gc.collect();torch.cuda.empty_cache()
                values=execute_oom_fallback(predict,restore,batch_state,oom_type=torch.cuda.OutOfMemoryError)
                outputs.append(values);order.extend(batch_rows.tolist())
    finally:
        if hasattr(iterator,'_shutdown_workers'):
            iterator._shutdown_workers()
    predictions,order=np.concatenate(outputs),np.array(order)
    if not np.array_equal(order,rows):
        raise ValueError('validation_alignment')
    return predictions,order

def _train():
    from .recipe import case_batches,finish_epoch,cv_metrics
    folder=OUT/f'fold-{FOLD}';folder.mkdir(exist_ok=False)
    train_rows,val_rows=np.flatnonzero(DATA['folds']!=FOLD),np.flatnonzero(DATA['folds']==FOLD)
    seed=SEEDS[FOLD];_seed(seed)
    model,optimizer,scaler=_model_optimizer()
    batch=dict(physical_batch=8,grad_accum=1,oom_fallback_used=False)
    total_steps=len(case_batches(train_rows))*20
    completed=0;timing=dict(validation_seconds=0.,write_seconds=0.);last_metrics=None
    try:
        for epoch in range(1,21):
            _check();began=time.monotonic();torch.cuda.reset_peak_memory_stats()
            train=_train_epoch(model,optimizer,scaler,train_rows,epoch,completed,total_steps,batch,folder)
            completed=train['completed_steps']
            def snapshot(actual_epoch):
                nonlocal last_metrics
                validation_start=time.monotonic()
                predictions,order=_predict(model,val_rows,folder,batch)
                last_metrics=cv_metrics(DATA['y_cv'][order],predictions,DATA['silent'][order],CONFIG['targets'])
                if last_metrics['denominator']!=12:
                    raise ValueError('UNDEFINED_TARGET_AUC')
                validation_seconds=time.monotonic()-validation_start
                writing_start=time.monotonic();prefix=f'epoch-{actual_epoch:02}'
                checkpoint=folder/(prefix+'.pt');oof=folder/(prefix+'-oof.private.npz')
                if checkpoint.exists() or oof.exists():
                    raise ValueError('F_SNAPSHOT_EXISTS')
                torch.save(dict(schema='F_FULL384_E20_RAW_V1',fold=FOLD,epoch=actual_epoch,model=model.state_dict(),
                    optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    scheduler=dict(completed_steps=completed,total_steps=total_steps),seed=seed,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),crop_mm=140,stored_mm=140,pixels=384,cache_slices=96,
                    train_windows=24,crop_pixels=384,crop_bounds=[0,384],resize=False,candidate=CANDIDATE,
                    initial_weight_sha256=WEIGHT_SHA,experiment_id=CONFIG['experiment_id'],evaluation_weights='RAW',
                    precision=PRECISION,batch=batch,bn_policy='frozen_running_stats_trainable_affine',
                    train_cases=len(train_rows),validation_rsna_only=True,oai_cases=0),checkpoint)
                np.savez_compressed(oof,study_uids=DATA['ids'][order],targets=np.array(CONFIG['targets']),
                    predictions=predictions,fold=np.array(FOLD),epoch=np.array(actual_epoch))
                hashes=dict(checkpoint_sha256=_sha(checkpoint),oof_sha256=_sha(oof))
                _save(folder/(prefix+'-snapshot.json'),dict(fold=FOLD,epoch=actual_epoch,cases=len(order),targets=12,
                    finite=True,checkpoint=checkpoint.name,oof=oof.name,**hashes,cv=last_metrics,
                    validation='Original_Faz0_with_silence'))
                _save(folder/(prefix+'-freeze.json'),dict(fold=FOLD,actual_epoch=actual_epoch,seed=seed,
                    cases=len(order),targets=12,**hashes,candidate=CANDIDATE,evaluation_weights='RAW',precision=PRECISION,
                    experiment_id=CONFIG['experiment_id'],initial_weight_sha256=WEIGHT_SHA,
                    private_input_hashes=CONFIG['private_inputs'],decision_rule_sha256=CONFIG['decision_rule_sha256'],
                    budget_amendment_sha256=CONFIG['budget_amendment_sha256'],
                    selection_decision_sha256=CONFIG['selection_evidence_hashes']['selection-evidence/decision.json']))
                _save(folder/'last-completed.json',dict(fold=FOLD,actual_epoch=actual_epoch,checkpoint=checkpoint.name,oof=oof.name))
                _copy_milestone([checkpoint,oof,folder/(prefix+'-snapshot.json'),folder/(prefix+'-freeze.json')],
                                f'fold-{FOLD}/{prefix}-drive-receipt.json')
                return dict(validation_seconds=validation_seconds,write_seconds=time.monotonic()-writing_start)
            decision=finish_epoch(epoch,began,DEADLINE,snapshot,timing)
            if epoch==1 and time.monotonic()-START+decision['projected_remaining_seconds']+300>CONFIG['max_projected_fold_seconds']:
                decision.update(stop=True,reason='F_ACTUAL_SPEED_GATE_STOP',budget_gate_pass=False)
            record=dict(fold=FOLD,epoch=epoch,**train,budget=decision,batch=dict(batch),
                precision=PRECISION,cv=last_metrics if decision['snapshot_saved'] else None,
                peak_vram_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_vram_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            _save(folder/f'epoch-{epoch:02}.json',record)
            if decision['stop']:
                result=dict(status='F_E20_COMPLETE' if epoch==20 else 'F_PARTIAL_E20_UNKNOWN',fold=FOLD,
                    actual_epoch=epoch,seed=seed,candidate=CANDIDATE,evaluation_weights='RAW',
                    experiment_id=CONFIG['experiment_id'],decision_rule_sha256=CONFIG['decision_rule_sha256'],
                    reason=decision['reason'],cv=last_metrics,denominator=last_metrics['denominator'],
                    macro=last_metrics['macro'],targets=last_metrics['targets'],batch=batch,precision=PRECISION,
                    elapsed_including_setup_seconds=time.monotonic()-START,further_training_started=False)
                _save(folder/'result.json',result);_save(OUT/'comparison.json',result)
                return result
    finally:
        del model,optimizer,scaler
        gc.collect();torch.cuda.empty_cache()
