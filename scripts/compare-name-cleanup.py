#!/usr/bin/env python3
"""Run an explicit whole-token name baseline against a frozen cleanup matrix."""
import argparse
import collections
import difflib
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import time

SPEC = importlib.util.spec_from_file_location('cleanup', Path(__file__).with_name('run-transcript-cleanup-benchmark.py'))
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def replace_names(text, aliases):
    """Case-sensitive whole-token edits only; preserve every other character."""
    if not aliases:
        return text, []
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in sorted(aliases, key=len, reverse=True)) + r')(?!\w)')
    edits = []
    def change(match):
        replacement = aliases[match[0]]
        if replacement != match[0]:
            edits.append({'start': match.start(), 'end': match.end(), 'before': match[0], 'after': replacement})
        return replacement
    return pattern.sub(change, text), edits


def alias_count(text, aliases):
    return sum(len(re.findall(r'(?<!\w)' + re.escape(a) + r'(?!\w)', text)) for a in aliases if aliases[a] != a)


def compare(run, config, output):
    manifest = json.loads((run / 'manifest.json').read_text())
    source = Path(manifest['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Source hash changed')
    aliases = config['aliases']
    segments = {s['id']: s for s in manifest['segments']}
    started = time.perf_counter()
    baseline, edits, candidates = {}, [], []
    for sid, segment in segments.items():
        baseline[sid], changes = replace_names(segment['text'], aliases)
        edits.extend(dict(e, segment=sid, timestamp=segment['timestamp']) for e in changes)
        for term in config.get('review_only', []):
            for match in re.finditer(r'(?<!\w)' + re.escape(term) + r'(?!\w)', segment['text']):
                candidates.append({'segment': sid, 'timestamp': segment['timestamp'], 'term': term,
                                   'context': segment['text'][max(0, match.start()-55):match.end()+65]})
    elapsed = time.perf_counter() - started
    groups = collections.defaultdict(list)
    for job in manifest['jobs']:
        response = run / 'responses' / (job['id'] + '.json')
        if response.exists():
            text = json.loads(response.read_text())['response']['message']['content']
            groups[(job['model'], job['variant'])].append((job['segment_id'], text))
    comparisons = []
    for (model, variant), rows in groups.items():
        source_aliases = sum(alias_count(segments[sid]['text'], aliases) for sid, _ in rows)
        remaining = sum(alias_count(text, aliases) for _, text in rows)
        # Normalize the configured name aliases on BOTH sides before counting other edits.
        other_edits = sum(cleanup.distance(cleanup.words(baseline[sid]), cleanup.words(replace_names(text, aliases)[0])) for sid, text in rows)
        comparisons.append({'model': model, 'prompt': variant, 'segments': len(rows),
                            'source_alias_occurrences': source_aliases, 'output_alias_occurrences': remaining,
                            'word_edits_after_name_normalization': other_edits})
    output.mkdir(parents=True, exist_ok=True)
    result = {'source_sha256': manifest['source_sha256'], 'config': config, 'segments': len(segments),
              'replacement_count': len(edits), 'processing_seconds': elapsed, 'edits': edits,
              'review_only_candidates': candidates, 'comparisons': comparisons,
              'accuracy': None, 'note': 'Explicit aliases are contextual assumptions, not audio-verified labels. Residual alias counts do not measure accuracy; deletion can lower them.'}
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / 'transcript.md').write_text('# Experimental name-only cleanup\n\nExplicit aliases; not audio-verified.\n\n' + '\n\n'.join('**'+s['timestamp']+'**\n\n'+baseline[sid] for sid,s in segments.items()))
    lines = ['# Name-only baseline comparison', '', f'{len(segments)} segments; {len(edits)} name substitutions; {len(candidates)} ambiguous occurrences left for review. No model or network required.', '',
             'Only exact, case-sensitive whole-token aliases change. All other characters are preserved. Alias choices were inspected in transcript context, not verified against audio. This is a glossary baseline, not a general spell checker.', '',
             '| Model / prompt | Saved segments | Input alias occurrences | Aliases still in output | Other word edits after name normalization |', '|---|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['model']} / {row['prompt']} | {row['segments']} | {row['source_alias_occurrences']} | {row['output_alias_occurrences']} | {row['word_edits_after_name_normalization']} |")
    lines += ['', 'The baseline replaces every configured alias and makes zero other character edits. LLM rows use only their available segments, so compare each with its own input count. Residual counts can fall through deletion, and other edits can be beneficial or harmful. Neither metric is WER, correction accuracy, or an entity identity verdict.', '', '## Substitutions', '']
    for (before, after), count in collections.Counter((e['before'], e['after']) for e in edits).items():
        lines.append(f'- `{before}` → `{after}`: {count}')
    lines += ['', '## Ambiguous occurrences left unchanged', '']
    for c in candidates:
        lines.append(f"- {c['timestamp']} / {c['term']}: {c['context']}")
    (output / 'summary.md').write_text('\n'.join(lines)+'\n')
    panels = []
    for sid, segment in segments.items():
        diff = difflib.HtmlDiff(wrapcolumn=85).make_table(segment['text'].splitlines(), baseline[sid].splitlines(), 'Original', 'Name-only', context=True)
        llms = ''.join('<details><summary>'+html.escape(model+' / '+variant)+'</summary><p>'+html.escape(text)+'</p></details>' for (model,variant), rows in groups.items() for key,text in rows if key==sid)
        panels.append('<section><h2>'+segment['timestamp']+' / '+sid+'</h2>'+diff+llms+'</section>')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Name-only comparison</title><style>body{font:16px system-ui;margin:24px;max-width:1500px}table{white-space:pre-wrap;width:100%;table-layout:fixed;font:14px monospace}td{overflow-wrap:anywhere}.diff_add{background:#bfeac5}.diff_sub{background:#ffcccc}.diff_chg{background:#ffed9c}section{border-top:1px solid #aaa;margin:24px 0}p{white-space:pre-wrap}</style><h1>Name-only baseline vs saved LLM outputs</h1><p>'+str(len(edits))+' explicit alias substitutions across '+str(len(segments))+' segments. Ambiguous ordinary words remain unchanged. Not audio-verified. See <a href="summary.md">summary</a> and <a href="results.json">exact rules and edit offsets</a>.</p>'+''.join(panels))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--aliases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run, json.loads(args.aliases.read_text()), args.output)
    print(f"{result['replacement_count']} substitutions across {result['segments']} segments; source preserved")
