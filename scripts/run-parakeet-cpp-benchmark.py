#!/usr/bin/env python3
"""On-device Parakeet benchmark (whisper.cpp ggml or LiteRT) on the five benchmark recordings.

Stages a ggml Parakeet model and the five WAVs into app-private storage, installs the
debug app and test APKs in place (never uninstalls), runs ParakeetCppBenchmarkTest,
pulls the raw per-window pieces, and merges/scores them on the host with the same
word merger and normalisation as the Mac FP32 reference runs.

Usage:
  scripts/run-parakeet-cpp-benchmark.py --model PATH.bin [--gpu] [--threads 4]
      [--no-gap-repair] [--skip-install] [--output DIR]
  scripts/run-parakeet-cpp-benchmark.py --model PATH.tflite [--gpu [--gpu-precision fp16]] ...
The engine follows the model extension (.bin = ggml, .tflite = LiteRT).
"""
import argparse,hashlib,json,pathlib,shlex,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'build/reports/parakeet-long-context-2026-11-16'))
from streaming_helpers import merge,text_of,word_error,words_from_pieces  # noqa: E402
PACKAGE='com.example.voicememo';TEST_PACKAGE=f'{PACKAGE}.test';SR=16000
RUNNER=f'{TEST_PACKAGE}/androidx.test.runner.AndroidJUnitRunner'
ENGINES={'.bin':('ggml','com.example.voicememo.transcription.ParakeetCppBenchmarkTest#transcribeStagedAudio','whispercpp-models','parakeet-cpp'),
 '.tflite':('litert','com.example.voicememo.transcription.ParakeetLiteRtBenchmarkTest#transcribeStagedAudio','parakeet-litert','parakeet-litert')}
VOCAB_JSON=pathlib.Path('/Volumes/X10 Pro/VoiceMemo/parakeet-fp32/nvidia/tokenizer.json')
SAMPLES=(  # same set as scripts/run-five-sample-precision-matrix.py
 ('controlled-german-170s','benchmark-inputs/whisper-suite/reference.wav','benchmark-inputs/whisper-suite/baseline.json',None),
 ('sample-test-153s','benchmark-inputs/sample-test/audio.wav','benchmark-inputs/sample-test/baseline.json',None),
 ('english-dialogue-short-9s','benchmark-inputs/article-samples/english-dialogue-short/audio.wav','benchmark-inputs/article-samples/english-dialogue-short/baseline.json','english-dialogue-short'),
 ('english-dialogue-proper-names-47s','benchmark-inputs/article-samples/english-dialogue-proper-names/audio.wav','benchmark-inputs/article-samples/english-dialogue-proper-names/baseline.json','english-dialogue-proper-names'),
 ('english-dialogue-mixed-fragments-51s','benchmark-inputs/article-samples/english-dialogue-mixed-fragments/audio.wav','benchmark-inputs/article-samples/english-dialogue-mixed-fragments/baseline.json','english-dialogue-mixed-fragments'),
)

def run(cmd,**kw):
 print('+ '+shlex.join(cmd),flush=True);return subprocess.run(cmd,check=True,**kw)
def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def device(wanted=None):
 out=subprocess.run(['adb','devices'],capture_output=True,text=True,check=True).stdout.splitlines()[1:]
 ok=[l.split()[0] for l in out if l.strip().endswith('\tdevice')]
 if wanted:
  if wanted not in ok:sys.exit(f'Requested device {wanted} is not connected: {out}')
  ok=[wanted]
 if len(ok)!=1:sys.exit(f'Expected exactly one authorized ADB device (or pass --serial), found: {out}')
 serial=ok[0];prop=lambda k:subprocess.run(['adb','-s',serial,'shell','getprop',k],capture_output=True,text=True).stdout.strip()
 if 'arm64-v8a' not in prop('ro.product.cpu.abilist'):
  sys.exit(f'Connected device {prop("ro.product.model")} is not arm64; VoiceMemo cannot run there.')
 installed=subprocess.run(['adb','-s',serial,'shell','pm','list','packages',PACKAGE],capture_output=True,text=True).stdout
 if f'package:{PACKAGE}\n' not in installed+'\n':
  sys.exit(f'VoiceMemo is not installed on {prop("ro.product.model")}; refusing to benchmark on an unexpected device.')
 print('device',serial,prop('ro.product.model'),flush=True)
 return serial
def stage(serial,local,relative):
 """Copy into app-private storage via /data/local/tmp and run-as; skip if the hash already matches."""
 want=sha256(local)
 have=subprocess.run(['adb','-s',serial,'shell','run-as',PACKAGE,'sha256sum',relative],capture_output=True,text=True)
 if have.returncode==0 and have.stdout.split()[:1]==[want]:print('staged (unchanged)',relative);return
 tmp=f'/data/local/tmp/{pathlib.Path(relative).name}'
 run(['adb','-s',serial,'push',str(local),tmp])
 run(['adb','-s',serial,'shell','run-as',PACKAGE,'mkdir','-p',str(pathlib.Path(relative).parent)])
 run(['adb','-s',serial,'shell',f'run-as {PACKAGE} sh -c "cat {tmp} > {relative}"'])
 run(['adb','-s',serial,'shell','rm','-f',tmp])
def piece_text(p,vocab):
 return p['text'] if p['text'] else vocab.get(p['id'],'')
def score(report,human):
 vocab={}
 if report.get('engine')=='litert':vocab={i:t for t,i in json.loads(VOCAB_JSON.read_text())['model']['vocab'].items()}
 rows=[];refs={x['id']:x['human_reference'] for x in human}
 by_audio={s['audio']:s for s in report['samples']}
 for sid,wav,base,hid in SAMPLES:
  s=by_audio[f'{sid}.wav'];words=[];prev=0
  for w in s['windows']:
   pieces=[{'text':piece_text(p,vocab).replace('▁',' '),'start_s':p['start_s'],'end_s':p['end_s']} for p in w['pieces']]
   if pieces:pieces[0]['text']=pieces[0]['text'].lstrip()
   words,_=merge(words,words_from_pieces(pieces),w['start_sample']/SR,prev/SR,'aligned');prev=w['end_sample']
  text=text_of(words);whisper=json.loads((ROOT/base).read_text())['text']
  rows.append({'sample':sid,'text':text,'vs_human':word_error(refs[hid],text) if hid else None,'vs_whisper_fp32':word_error(whisper,text),
   'rtf':s['rtf'],'wall_ms':s['wall_ms'],'decodes':s['decodes'],'gaps_checked':s['gaps_checked'],'gaps_repaired':s['gaps_repaired']})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=pathlib.Path);ap.add_argument('--gpu',action='store_true');ap.add_argument('--gpu-precision',choices=['fp32','fp16'],default='fp32')
 ap.add_argument('--threads',type=int,default=4);ap.add_argument('--no-gap-repair',action='store_true');ap.add_argument('--skip-install',action='store_true');ap.add_argument('--serial');ap.add_argument('--repair-mode',choices=['tight','full'],default='tight')
 ap.add_argument('--output',type=pathlib.Path,default=ROOT/'build/reports/parakeet-cpp-device');a=ap.parse_args()
 engine,test,model_dir,report_dir=ENGINES[a.model.suffix]
 serial=device(a.serial);run_id=time.strftime('%Y%m%dT%H%M%S')
 if not a.skip_install:
  apk=ROOT/'app/build/outputs/apk';run(['adb','-s',serial,'install','-r',str(apk/'debug/app-debug.apk')]);run(['adb','-s',serial,'install','-r',str(apk/'androidTest/debug/app-debug-androidTest.apk')])
 stage(serial,a.model,f'files/{model_dir}/{a.model.name}')
 if a.model.name.endswith('_encoder.tflite'):  # split LiteRT layout: decoder file sits next to the encoder
  dec=a.model.with_name(a.model.name.replace('_encoder.tflite','_decoder.tflite'));stage(serial,dec,f'files/{model_dir}/{dec.name}')
 names=[]
 for sid,wav,_,_ in SAMPLES:stage(serial,ROOT/wav,f'files/gpu-audio/{sid}.wav');names.append(f'{sid}.wav')
 run(['adb','-s',serial,'shell','am','instrument','-w','-r','-e','class',test,'-e','parakeet_model_name',a.model.name,'-e','parakeet_audio_names',','.join(names),
  '-e','parakeet_use_gpu',str(a.gpu).lower(),'-e','parakeet_backend','gpu' if a.gpu else 'cpu','-e','parakeet_gpu_precision',a.gpu_precision,'-e','parakeet_threads',str(a.threads),'-e','parakeet_gap_repair',str(not a.no_gap_repair).lower(),'-e','parakeet_repair_mode',a.repair_mode,'-e','run_id',run_id,RUNNER])
 a.output.mkdir(parents=True,exist_ok=True);raw=a.output/f'parakeet-{run_id}.json'
 with raw.open('wb') as f:subprocess.run(['adb','-s',serial,'exec-out','run-as',PACKAGE,'cat',f'cache/{report_dir}/parakeet-{run_id}.json'],stdout=f,check=True)
 if engine=='litert':
  with (a.output/f'features-{run_id}.f32').open('wb') as f:subprocess.run(['adb','-s',serial,'exec-out','run-as',PACKAGE,'cat',f'cache/{report_dir}/features-{run_id}.f32'],stdout=f,check=True)
 report=json.loads(raw.read_text());human=json.loads((ROOT/'blog/data/2026-07-27-three-model-comparison.json').read_text())['examples']
 rows=score(report,human);eh=sum(r['vs_human']['edits'] for r in rows if r['vs_human'])
 ew=sum(r['vs_whisper_fp32']['edits'] for r in rows);nw=sum(r['vs_whisper_fp32']['reference_words'] for r in rows)
 audio=sum(s['audio_seconds'] for s in report['samples']);wall=sum(s['wall_ms'] for s in report['samples'])/1000
 summary={'run_id':run_id,'engine':engine,'model':report['model'],'backend':report['backend'],'threads':report['threads'],'gap_repair':report['gap_repair'],'repair_mode':report.get('repair_mode','full'),
  'human_errors':eh,'human_words':187,'whisper_fp32_edits':ew,'whisper_fp32_words':nw,'aggregate_rtf':wall/audio,'inference_wall_s':wall,
  'model_load_ms':report['model_load_ms'],'peak_total_pss_gib':report['peak_total_pss_kib']/2**20,'rows':rows}
 (a.output/f'summary-{run_id}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=1))
 for r in rows:print(r['sample'],'human',r['vs_human'] and r['vs_human']['edits'],'whisper',r['vs_whisper_fp32']['edits'],'rtf',round(r['rtf'],3),'gaps',r['gaps_checked'],r['gaps_repaired'])
if __name__=='__main__':main()
