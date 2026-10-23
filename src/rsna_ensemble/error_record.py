"""Bounded, fail-closed notebook diagnostics; never serialize locals or source.

Only explicitly registered source labels, stages and diagnostic codes are
retained verbatim. Unknown message tokens are redacted, not guessed safe.
"""
import re
import traceback
from contextlib import contextmanager
import json
from pathlib import Path

_WORDS = frozenset('No such file or directory not found cannot import name from '
    'module named No module object has no attribute expected got missing '
    'size mismatch shape dimension dimensions tensor tensors input output '
    'CUDA out of memory allocation failed error invalid argument arguments '
    'Permission denied Access denied RuntimeError ValueError TypeError '
    'FileNotFoundError ModuleNotFoundError KeyError IndexError AssertionError '
    'TimeoutError OSError StopIteration incompatible keys state_dict Unexpected '
    'Missing invalid literal for int with base could convert string to float '
    'unpack too many values enough unsupported operand type types for '
    'division by zero list index range must be bytes like str NoneType '
    'unexpected keyword positional required callable bool integer Tensor '
    'mat and shapes multiplied device devices cpu cuda torch torch.Size '
    'same Expected found but was is a an the in on at as with and or of '
    'closed read write file timed timeout process exited nonzero killed '
    'hash checksum schema order finite malformed truncated archive'.split())


def masked_message(exc, *, allowed_codes=()):
    try:
        text = str(exc)[:4096]
    except BaseException:
        return '<message unavailable: exception string failed>'
    text = re.sub(r'https?://\S+', '<redacted>', text)
    text = re.sub(r"(['\"]).*?\1", '<redacted>', text, flags=re.S)
    text = re.sub(r'(?:[A-Za-z]:[\\/]|/)[^\s,;]+', '<redacted>', text)
    tokens = re.findall(r'[A-Za-z_][A-Za-z_0-9.]*|\d+(?:\.\d+)*|[^\w\s]', text)
    safe = [t if t in _WORDS or t in allowed_codes or t in ':,()[]=-' else '<redacted>' for t in tokens]
    message = ' '.join(safe)
    message = re.sub(r'(?:<redacted>\s*)+', '<redacted> ', message).strip()
    return message[:512] or '<empty message>'


def capture_error(exc, *, stage, allowed_files, allowed_codes=(), _seen=()):
    if not re.fullmatch(r'[A-Z][A-Z_]{1,63}', stage):
        raise ValueError('ERROR_RECORD_STAGE_NOT_REGISTERED')
    frames = []
    for frame in traceback.extract_tb(exc.__traceback__, limit=32):
        function = frame.name if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,79}|<module>', frame.name) else '<masked-function>'
        frames.append({'file': allowed_files.get(frame.filename, '<external-file>'),
                       'line': frame.lineno, 'function': function})
    name = type(exc).__name__
    name = name if re.fullmatch(r'[A-Za-z_]{1,80}', name) else '<masked-exception-type>'
    result = {'schema': 'MASKED_NOTEBOOK_ERROR_V1', 'exception_type': name,
            'message_masked': masked_message(exc, allowed_codes=allowed_codes),
            'stage': stage, 'traceback': frames,
            'traceback_available': bool(frames), 'locals_recorded': False,
            'message_policy': 'ALLOWLIST_TOKENS_NO_VALUES'}
    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    if cause is not None and id(cause) not in (*_seen, id(exc)) and len(_seen) < 3:
        result['cause'] = capture_error(cause, stage=stage, allowed_files=allowed_files,
                                       allowed_codes=allowed_codes, _seen=(*_seen, id(exc)))
    return result


@contextmanager
def record_failure(path, *, stage, allowed_files, allowed_codes=()):
    """Write once; preserve the original exception even if logging fails."""
    try:
        yield
    except BaseException as exc:
        record = capture_error(exc, stage=stage, allowed_files=allowed_files,
                               allowed_codes=allowed_codes)
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as handle:
                json.dump(record, handle, allow_nan=False, sort_keys=True)
                handle.write('\n')
        except OSError:
            # This line is already sanitized, never repr(exc) or a traceback.
            print(json.dumps({'diagnostic_write_failed': True, 'error': record}), flush=True)
        raise


def install_child_hook(directory, *, stage, allowed_files, allowed_codes=()):
    """Record uncaught child failures without invoking the raw stderr hook."""
    import os
    import sys
    def hook(kind, exc, tb):
        record = capture_error(exc.with_traceback(tb), stage=stage,
                               allowed_files=allowed_files, allowed_codes=allowed_codes)
        path = Path(directory) / f'child-error-{os.getpid()}.json'
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as handle:
                json.dump(record, handle, allow_nan=False, sort_keys=True)
                handle.write('\n')
        except OSError:
            print(json.dumps({'diagnostic_write_failed': True, 'error': record}), flush=True)
    sys.excepthook = hook
