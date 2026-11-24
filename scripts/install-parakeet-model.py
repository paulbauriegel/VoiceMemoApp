#!/usr/bin/env python3
"""Install the Parakeet TDT 0.6B v3 LiteRT model into the VoiceMemo app's private storage.

The 1.3 GB split model (FP16-safe-LayerNorm encoder + predict/joint decoder) does not fit next
to Whisper inside a 4 GiB APK, so it is side-installed into files/parakeet-litert/ of the
installed app. The app is never reinstalled or cleared; existing recordings are untouched.

  scripts/install-parakeet-model.py [--source DIR] [--serial SERIAL]

Files whose size and SHA-256 already match on the device are skipped.
"""
import argparse
import hashlib
import pathlib
import shlex
import subprocess
import sys

PACKAGE = 'com.example.voicememo'
DEVICE_DIRECTORY = 'files/parakeet-litert'
DEFAULT_SOURCE = pathlib.Path('/Volumes/X10 Pro/VoiceMemo/parakeet-litert')
FILES = {  # must match ParakeetAssets in TranscriptionModel.kt
    'parakeet_tdt_0.6b_v3_30s_fp16_lnsafe_encoder.tflite':
        (1_245_335_280, 'a9f81d5dc0a46241f45f0d3103db1d7552dae37e1f6a81c2014eff81278187ea'),
    'parakeet_tdt_0.6b_v3_30s_fp16_lnsafe_decoder.tflite':
        (69_903_132, '575417424ca4d55dee5ec918044debf852fe049b806b2dcac3453b14ad4f7857'),
}


def adb(serial, *args, check=True):
    command = ['adb'] + (['-s', serial] if serial else []) + list(args)
    return subprocess.run(command, check=check, capture_output=True, text=True)


def run_as(serial, script, check=True):
    return adb(serial, 'shell', f'run-as {PACKAGE} sh -c {shlex.quote(script)}', check=check)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument('--serial')
    args = parser.parse_args()

    if run_as(args.serial, 'true', check=False).returncode != 0:
        sys.exit(f'{PACKAGE} is not installed as a debuggable app on the device; install the APK first.')
    run_as(args.serial, f'mkdir -p {DEVICE_DIRECTORY}')
    changed = False
    for name, (size, expected) in FILES.items():
        target = f'{DEVICE_DIRECTORY}/{name}'
        present = run_as(args.serial, f'stat -c %s {target} 2>/dev/null && sha256sum {target}', check=False).stdout.split()
        if len(present) >= 2 and present[0] == str(size) and present[1] == expected:
            print(f'{name}: already installed')
            continue
        local = args.source / name
        if not local.is_file() or local.stat().st_size != size:
            sys.exit(f'{local} is missing or has the wrong size.')
        print(f'{name}: verifying local copy')
        if sha256(local) != expected:
            sys.exit(f'{local} does not match the pinned SHA-256.')
        staging = f'/data/local/tmp/{name}'
        print(f'{name}: pushing {size / 1e6:.0f} MB')
        adb(args.serial, 'push', str(local), staging)
        try:
            run_as(args.serial, f'cp {staging} {target}.tmp && mv {target}.tmp {target}')
        finally:
            adb(args.serial, 'shell', 'rm', '-f', staging, check=False)
        installed = run_as(args.serial, f'sha256sum {target}').stdout.split()[0]
        if installed != expected:
            sys.exit(f'{name}: SHA-256 on the device does not match after copying.')
        print(f'{name}: installed')
        changed = True
    if changed:  # the app re-hashes on its next load
        run_as(args.serial, f'rm -f {DEVICE_DIRECTORY}/.verified', check=False)
    print('Parakeet model ready.')


if __name__ == '__main__':
    main()
