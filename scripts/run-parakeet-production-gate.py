#!/usr/bin/env python3
"""Five-sample device gate for the shipped Parakeet path (ParakeetProductionGateTest).

Installs the debug + androidTest APKs in place (never uninstalls), stages the five gate WAVs into
app-private storage, runs the production window source + engine + word merge on the phone, and
scores the app's own merged text against the human references and the pinned Whisper FP32
transcripts, exactly like scripts/run-parakeet-cpp-benchmark.py.

  scripts/run-parakeet-production-gate.py [--serial SERIAL] [--skip-install] [--cpu]
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bench', ROOT / 'scripts/run-parakeet-cpp-benchmark.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

TEST = 'com.example.voicememo.transcription.ParakeetProductionGateTest#transcribeStagedAudioLikeTheApp'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial')
    parser.add_argument('--skip-install', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=pathlib.Path, default=ROOT / 'build/reports/parakeet-production-gate')
    args = parser.parse_args()
    serial = bench.device(args.serial)
    run_id = time.strftime('%Y%m%dT%H%M%S')
    if not args.skip_install:
        apk = ROOT / 'app/build/outputs/apk'
        bench.run(['adb', '-s', serial, 'install', '-r', str(apk / 'debug/app-debug.apk')])
        bench.run(['adb', '-s', serial, 'install', '-r', str(apk / 'androidTest/debug/app-debug-androidTest.apk')])
    names = []
    for sample_id, wav, _, _ in bench.SAMPLES:
        bench.stage(serial, ROOT / wav, f'files/gpu-audio/{sample_id}.wav')
        names.append(f'{sample_id}.wav')
    bench.run(['adb', '-s', serial, 'shell', 'am', 'instrument', '-w', '-r', '-e', 'class', TEST,
               '-e', 'parakeet_audio_names', ','.join(names), '-e', 'parakeet_backend', 'cpu' if args.cpu else 'gpu',
               '-e', 'run_id', run_id, bench.RUNNER])
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / f'gate-{run_id}.json'
    with raw.open('wb') as handle:
        subprocess.run(['adb', '-s', serial, 'exec-out', 'run-as', bench.PACKAGE, 'cat', f'cache/parakeet-gate/{run_id}.json'],
                       stdout=handle, check=True)
    report = json.loads(raw.read_text())
    human = {x['id']: x['human_reference'] for x in
             json.loads((ROOT / 'blog/data/2026-07-27-three-model-comparison.json').read_text())['examples']}
    by_audio = {s['audio']: s for s in report['samples']}
    rows = []
    for sample_id, _, baseline, human_id in bench.SAMPLES:
        sample = by_audio[f'{sample_id}.wav']
        whisper = json.loads((ROOT / baseline).read_text())['text']
        rows.append({
            'sample': sample_id, 'rtf': sample['rtf'], 'wall_ms': sample['wall_ms'], 'text': sample['text'],
            'vs_human': bench.word_error(human[human_id], sample['text']) if human_id else None,
            'vs_whisper_fp32': bench.word_error(whisper, sample['text']),
        })
    audio = sum(s['audio_seconds'] for s in report['samples'])
    wall = sum(s['wall_ms'] for s in report['samples']) / 1000
    summary = {
        'run_id': run_id, 'backend': report['backend'], 'model_load_ms': report['model_load_ms'],
        'human_errors': sum(r['vs_human']['edits'] for r in rows if r['vs_human']), 'human_words': 187,
        'whisper_fp32_edits': sum(r['vs_whisper_fp32']['edits'] for r in rows),
        'whisper_fp32_words': sum(r['vs_whisper_fp32']['reference_words'] for r in rows),
        'aggregate_rtf': wall / audio, 'inference_wall_s': wall,
        'memory_after_load': report['memory_after_load'], 'rows': rows,
    }
    (args.output / f'summary-{run_id}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))
    for row in rows:
        print(row['sample'], 'human', row['vs_human'] and row['vs_human']['edits'],
              'whisper', row['vs_whisper_fp32']['edits'], 'rtf', round(row['rtf'], 3))


if __name__ == '__main__':
    main()
