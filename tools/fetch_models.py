"""Prepare model files automatically, without redistributing any of them.

    .venv/bin/python tools/fetch_models.py            # ExMR (default model) + person detection
    .venv/bin/python tools/fetch_models.py --check    # only report what is ready

Every file comes from its official source, under the user's own agreement:

* Public weights (ExMR, Sapiens2, HaMeR, YOLO, OSNet) download from their hosts.
  Hugging Face files use huggingface_hub (Xet transfers, resumable).
* SAM 3D Body is gated: request access once at
  https://huggingface.co/facebook/sam-3d-body-dinov3 and run `hf auth login`.
* SMPL-X and MANO need a free account at smpl-x.is.tue.mpg.de / mano.is.tue.mpg.de.
  Their licenses forbid redistribution, so they are never bundled. This script
  1) imports archives or folders you already downloaded (~/Downloads, or --from), or
  2) downloads them from the official server with your account. The password is
     asked for interactively (or $MPI_PASSWORD) and is never stored.

Every file is registered through the app's own import, which checks size and hash.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('RGB_MOCAP_DATA', str(ROOT))
from mocap import models  # noqa: E402

# key -> (repo, filename, repo_type, revision)
HUGGING_FACE = {
    'exmr': ('twehrbein/ExMR', 'exmr.safetensors', 'model', models.EXMR_REV),
    'exmr_config': ('twehrbein/ExMR', 'config.json', 'model', models.EXMR_REV),
    'exmr_sapiens': ('facebook/sapiens2-pose-1b', 'sapiens2_1b_pose.safetensors', 'model',
                     'f5fed8b97b99698d5eea1d14ff0855d0b4c3f000'),
    'hand_hamer': ('geopavlakos/HaMeR', '_DATA/hamer_ckpts/checkpoints/hamer.ckpt', 'space', 'main'),
    'flow_sam': ('facebook/sam-3d-body-dinov3', 'model.ckpt', 'model', 'main'),
    'flow_sam_config': ('facebook/sam-3d-body-dinov3', 'model_config.yaml', 'model', 'main'),
    'flow_mhr': ('facebook/sam-3d-body-dinov3', 'assets/mhr_model.pt', 'model', 'main'),
}
GATED = {'flow_sam', 'flow_sam_config', 'flow_mhr'}
# Registered body/hand models: official archive, and the file inside it that the app imports.
REGISTERED = {
    'smplx': dict(domain='smplx', archive='models_smplx_v1_1.zip', member='models/smplx/SMPLX_NEUTRAL.npz',
                  site='https://smpl-x.is.tue.mpg.de/'),
    'mano_right': dict(domain='mano', archive='mano_v1_2.zip', member='mano_v1_2/models/MANO_RIGHT.pkl',
                       site='https://mano.is.tue.mpg.de/'),
}
PROFILES = {'exmr': ['yolo', 'person_reid', 'exmr', 'exmr_config', 'exmr_sapiens', 'hand_hamer', 'hand_mean',
                     'flow_sam', 'flow_sam_config', 'flow_mhr', 'mano_right', 'smplx']}


def ready(key):
    return models.ready(models.BY_KEY[key])


def say(*parts):
    print(*parts, flush=True)


def fetch_hugging_face(key, work):
    from huggingface_hub import hf_hub_download
    repo, name, kind, revision = HUGGING_FACE[key]
    for attempt in range(1, 21):
        try:
            return Path(hf_hub_download(repo, name, repo_type=kind, revision=revision, local_dir=work / key))
        except Exception as error:  # network drops: resume
            text = str(error)
            if key in GATED and ('401' in text or '403' in text or 'gated' in text.lower()):
                raise RuntimeError('SAM 3D Body is gated: request access at '
                                   'https://huggingface.co/facebook/sam-3d-body-dinov3, then run `hf auth login`.')
            say(f'  {key}: retry {attempt} after {type(error).__name__}')
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f'{key}: download kept failing')


def find_local(spec, extra):
    """An official archive or extracted folder the user already downloaded."""
    folders = [Path(p).expanduser() for p in extra] + [Path.home() / 'Downloads', Path.home() / 'Desktop']
    member = Path(spec['member'])
    for folder in folders:
        if not folder.exists():
            continue
        archive = folder / spec['archive']
        if archive.is_file():
            return archive
        for base in (folder, folder.parent):
            # Extracted copies: <folder>/<member> or the member's tail (models/smplx/...)
            for candidate in (base / member, base / Path(*member.parts[1:])):
                if candidate.is_file():
                    return candidate
        for hit in folder.glob('**/' + member.name):
            if hit.is_file() and hit.parent.name == member.parent.name:
                return hit
    return None


def download_registered(spec, work, email):
    import requests
    password = os.environ.get('MPI_PASSWORD') or getpass.getpass(f'  Password for {email} at {spec["site"]} (not stored): ')
    url = f'https://download.is.tue.mpg.de/download.php?domain={spec["domain"]}&resume=1&sfile={spec["archive"]}'
    target = work / spec['archive']
    with requests.post(url, data={'username': email, 'password': password}, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if 'zip' not in response.headers.get('content-type', '') and 'octet' not in response.headers.get('content-type', ''):
            raise RuntimeError(f'Sign-in failed or the license is not accepted for this account. Log in at {spec["site"]}, '
                               'accept the license, and confirm the activation email first.')
        with open(target, 'wb') as out:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
    return target


def extract_member(path, spec, work):
    if path.suffix.lower() != '.zip':
        return path
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(Path(spec['member']).name) and
                 Path(n).parent.name == Path(spec['member']).parent.name]
        if not names:
            raise RuntimeError(f'{path.name} does not contain {spec["member"]}')
        return Path(archive.extract(names[0], work))


def register(key, path):
    final = models.import_body_model(path) if key in ('smplx', 'smpl') else models.import_model_file(key, path)
    say(f'  ready: {models.BY_KEY[key].name} -> {final}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--profile', default='exmr', choices=sorted(PROFILES))
    parser.add_argument('--from', dest='sources', action='append', default=[],
                        help='folder holding downloaded SMPL-X/MANO archives (default: ~/Downloads, ~/Desktop)')
    parser.add_argument('--mpi-email', default=os.environ.get('MPI_EMAIL'),
                        help='account email for smpl-x/mano.is.tue.mpg.de, to download missing body models')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    keys = PROFILES[args.profile]
    missing = [k for k in keys if not ready(k)]
    say(f'{len(keys) - len(missing)}/{len(keys)} model files ready for {args.profile}.')
    if args.check or not missing:
        for k in missing:
            say('  missing:', models.BY_KEY[k].name)
        return 0 if not missing else 1
    work = Path(tempfile.mkdtemp(prefix='rgbmocap-models-', dir=ROOT / 'cache' if (ROOT / 'cache').exists() else None))
    problems = []
    try:
        for key in missing:
            name = models.BY_KEY[key].name
            say(f'- {name}')
            try:
                if key in REGISTERED:
                    spec = REGISTERED[key]
                    found = find_local(spec, args.sources)
                    if found is None and args.mpi_email:
                        say(f'  downloading {spec["archive"]} with your account')
                        found = download_registered(spec, work, args.mpi_email)
                    if found is None:
                        raise RuntimeError(f'register at {spec["site"]}, download {spec["archive"]} into ~/Downloads '
                                           f'(or pass --mpi-email to download it here), then rerun.')
                    say(f'  using {found}')
                    register(key, extract_member(found, spec, work))
                elif key in HUGGING_FACE:
                    register(key, fetch_hugging_face(key, work))
                else:
                    models.download(key)
                    say('  ready:', name)
            except Exception as error:
                problems.append(f'{name}: {error}')
                say('  not ready:', error)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if problems:
        say('\nStill needed:'); [say(' -', p) for p in problems]
        return 1
    say('\nAll model files are ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
