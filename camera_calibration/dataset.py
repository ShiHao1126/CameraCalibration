"""Append-only raw logs and atomic derived artifacts under outputs/."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import uuid
from .models import dumps, jsonable


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:
        json.dump(jsonable(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    out = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'Incomplete/corrupt log {path}:{line_number}') from e
    return out


class Journal:
    def __init__(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', buffering=1)

    def append(self, value):
        self.file.write(dumps(value)+'\n')

    def close(self):
        if not self.file.closed:
            self.file.flush(); os.fsync(self.file.fileno()); self.file.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def create_session(cfg, kind='session'):
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    path = Path(cfg['outputs']) / f"{stamp}_cam-{cfg['camera']['serial']}_{kind}"
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'manifest.json', {'schema_version':1, 'created_at':datetime.now().astimezone().isoformat(),
               'kind':kind, 'config':cfg, 'camera_serial':cfg['camera']['serial'],
               'robot_serial':cfg['robot']['serial'], 'reference_frame':'world', 'status':'created'})
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def update_manifest(session, **fields):
    p = Path(session)/'manifest.json'
    value = read_json(p); value.update(fields); write_json(p, value)
