import math, random, json, time, gc
from pathlib import Path
import numpy as np
import torch
from .recipe import *
from rsna_ensemble.local_runtime import json_save, save_failure, diagnostic
class SafeCases(CorpusCases):
    def __init__(self, *args, error_folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_folder = Path(error_folder)

    def __getitem__(self, i):
        try:
            return super().__getitem__(i)
        except BaseException as exc:
            record = diagnostic(exc, 'A_CORPUS_GETITEM')
            try:
                json_save(self.error_folder / f'data-error-{os.getpid()}.json', record)
            except OSError:
                # Source exception is still propagated; no raw data is printed.
                print(json.dumps({'diagnostic_write_failed': True, 'error': record}), flush=True)
            raise

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)

def predict(model, loader):
    model.eval()
    outputs, order = [], []
    iterator = iter(loader)
    try:
        with torch.inference_mode():
            for x, valid, planes, rows in iterator:
                check_deadline()
                x, valid, planes = x.cuda(non_blocking=True), valid.cuda(non_blocking=True), planes.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    first = model(x, valid, torch.zeros(x.shape[1], dtype=torch.long, device=x.device))
                    mirrored = mirror_windows(x.flatten(0, 1), planes.flatten()).reshape_as(x)
                    second = model(mirrored, valid, torch.zeros(x.shape[1], dtype=torch.long, device=x.device))
                    predictions = (first.float().sigmoid() + second.float().sigmoid()) / 2
                if not torch.isfinite(predictions).all():
                    raise ValueError('nonfinite_prediction')
                outputs.append(predictions.cpu().numpy())
                order.extend(rows.tolist())
    finally:
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    return np.concatenate(outputs), np.array(order)

def train_fold(fold):
    if fold not in range(5):
        raise ValueError('AUTHORIZED_FOLD_REQUIRED')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('COLAB_A100_REQUIRED')
    check_deadline()
    folder = OUT/f'fold-{fold}'
    folder.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    seed = SEEDS[fold]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    train_rows, val_rows, plan = fold_rows(DATA, fold)
    from safetensors.torch import load_file
    state = load_file(str(WEIGHT_PATH), device='cpu')
    model = ConvNeXtMIL(state).cuda(); del state
    optimizer = torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=2e-4),
        dict(params=list(model.projection.parameters()) + list(model.mil.parameters()), lr=1e-3)], weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    train = SafeCases(DATA['volumes'], DATA['masks'], DATA['mapping'], train_rows,
                      training=True, error_folder=folder)
    val = SafeCases(DATA['volumes'], DATA['masks'], DATA['mapping'], val_rows,
                    training=False, error_folder=folder)
    val_loader = torch.utils.data.DataLoader(val, batch_size=8, shuffle=False, num_workers=4,
                                            pin_memory=True, worker_init_fn=seed_worker)
    total_steps, completed, skip_count, consecutive = plan['scheduler_total_steps'], 0, 0, 0
    timing = {'validation_seconds': 0., 'write_seconds': 0.}
    last_metrics = None
    try:
        for epoch in range(1, 21):
            check_deadline()
            epoch_start = time.monotonic()
            model.train()
            shuffled = np.random.default_rng(seed + epoch).permutation(len(train)).tolist()
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = torch.utils.data.DataLoader(train, batch_sampler=case_batches(shuffled), num_workers=8,
                      pin_memory=True, worker_init_fn=seed_worker, generator=generator)
            losses = []
            iterator = iter(loader)
            try:
                for x, valid, planes, rows in iterator:
                    check_deadline()
                    optimizer.zero_grad(set_to_none=True)
                    factor = lr_multiplier(completed, total_steps)
                    for group, base in zip(optimizer.param_groups, (2e-4, 1e-3)):
                        group['lr'] = base * factor
                    x, valid = x.cuda(non_blocking=True), valid.cuda(non_blocking=True)
                    target = torch.from_numpy(DATA['y'][rows.numpy()].astype(np.float32)).cuda()
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(x, valid, torch.zeros(24, dtype=torch.long, device=x.device))
                        loss = soft_bce(logits.float(), target)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite_loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradients_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
                    scaler.step(optimizer); scaler.update()
                    completed += 1
                    consecutive = 0 if gradients_finite else consecutive + 1
                    skip_count += int(not gradients_finite)
                    if consecutive > 8 or scaler.get_scale() < 1:
                        raise ValueError('nonfinite_gradient_stop')
                    losses.append(float(loss.detach()))
                    if time.monotonic()-epoch_start > 3600:
                        raise TimeoutError('epoch_train_limit')
            finally:
                if hasattr(iterator, '_shutdown_workers'):
                    iterator._shutdown_workers()
            train_seconds = time.monotonic()-epoch_start

            def snapshot(actual_epoch):
                nonlocal last_metrics
                validation_start = time.monotonic()
                predictions, order = predict(model, val_loader)
                if not np.array_equal(order, val_rows):
                    raise ValueError('validation_alignment')
                last_metrics = cv_metrics(DATA['y_cv'][order], predictions, DATA['silent'][order], TARGETS)
                validation_seconds = time.monotonic()-validation_start
                writing_start = time.monotonic()
                checkpoint = dict(schema='A_BIG_COLAB_CHECKPOINT_V1', fold=fold, epoch=actual_epoch,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    scheduler=dict(completed_steps=completed, total_steps=total_steps), seed=seed,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), crop_mm=120, stored_mm=140, pixels=320, cache_slices=96, train_windows=24, crop_pixels=329,
                    initial_weight_sha256=WEIGHT_SHA, experiment_id=CONFIG['experiment_id'], **CONFIG['checkpoint_lineage'],
                    train_cases=len(train_rows), validation_rsna_only=True, oai_cases=0)
                prefix = f'epoch-{actual_epoch:02}'
                torch.save(checkpoint, folder/f'{prefix}.pt')
                np.savez_compressed(folder/f'{prefix}-oof.private.npz', study_uids=DATA['ids'][order],
                    targets=np.array(TARGETS), predictions=predictions, fold=np.array(fold), epoch=np.array(actual_epoch))
                checkpoint_sha = sha_file(folder/f'{prefix}.pt')
                oof_sha = sha_file(folder/f'{prefix}-oof.private.npz')
                json_save(folder/f'{prefix}-snapshot.json', dict(fold=fold, epoch=actual_epoch,
                    cases=len(order), targets=12, finite=True, checkpoint=f'{prefix}.pt',
                    checkpoint_sha256=checkpoint_sha, oof_sha256=oof_sha,
                    oof=f'{prefix}-oof.private.npz', validation='Original_Faz0_with_silence', cv=last_metrics))
                json_save(folder/f'{prefix}-freeze.json', dict(fold=fold, actual_epoch=actual_epoch, seed=seed,
                    cases=len(order), targets=12, oof_sha256=oof_sha, checkpoint_sha256=checkpoint_sha,
                    experiment_id=CONFIG['experiment_id'], initial_weight_sha256=WEIGHT_SHA, **CONFIG['checkpoint_lineage'],
                    private_input_hashes=CONFIG['private_inputs']))
                json_save(folder/'last-completed.json', dict(fold=fold, actual_epoch=actual_epoch,
                    checkpoint=f'{prefix}.pt', oof=f'{prefix}-oof.private.npz'))
                return dict(validation_seconds=validation_seconds, write_seconds=time.monotonic()-writing_start)

            decision = finish_epoch(epoch, epoch_start, WORK_DEADLINE, snapshot, timing)
            record = dict(fold=fold, epoch=epoch, train_seconds=train_seconds, completed_steps=completed,
                          skip_count=skip_count, mean_train_loss=float(np.mean(losses)), budget=decision,
                          nominal_batch_size=8, batch_reduced=False,
                          peak_vram_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_vram_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          cv=last_metrics if decision['snapshot_saved'] else None)
            json_save(folder/f'epoch-{epoch:02}.json', record)
            print(json.dumps({'fold': fold, 'actual_epoch': epoch, 'train_seconds': round(train_seconds, 2),
                              'macro': record['cv']['macro'] if record['cv'] else None, 'reason': decision['reason']}))
            if decision['stop']:
                result = dict(fold=fold, actual_epoch=epoch, status='E20_COMPLETE' if epoch == 20 else 'PARTIAL_E20_UNKNOWN',
                              reason=decision['reason'], training_plan=plan, cv=last_metrics,
                              elapsed_including_setup_seconds=time.monotonic()-STARTED)
                json_save(folder/'result.json', result)
                return result
    finally:
        del model, optimizer, scaler, train, val, val_loader
        gc.collect()
        torch.cuda.empty_cache()
