#!/usr/bin/env python3
"""Export Parakeet TDT 0.6B v3 to LiteRT with a 30-second encoder and a one-step decoder.

Signatures (one .tflite):
  encode(features[1,FRAMES,128] f32, mask[1,FRAMES] i32) -> encoded[1,T,640] f32
      (projected encoder frames; T = 375 for 30 s)
  decode_step(token[1,1] i32, h[2,1,640] f32, c[2,1,640] f32, frame[1,640] f32)
      -> logits[1,8198] f32, h'[2,1,640], c'[2,1,640]

The decoder emits one joint row per step instead of the whole [T, slots, 8198] block of
the litert-community export, so per-step output stays at ~33 KB for any window length.

Run with the litert-export venv (litert-torch, torch 2.14, transformers 5.23):
  python scripts/export-parakeet-litert-30s.py --check-only
  python scripts/export-parakeet-litert-30s.py --quant int8 --out DIR
Fine-tunes with the same architecture (e.g. primeline/parakeet-primeline saved in Transformers format):
  python scripts/export-parakeet-litert-30s.py --model-dir DIR --name parakeet_primeline --source-revision REV \
      --split --fp16-safe-layernorm --quant fp16
"""
import argparse,json,pathlib,sys,time,wave
import numpy as np
import torch
from transformers import AutoModelForTDT,AutoProcessor
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR=pathlib.Path('/Volumes/X10 Pro/VoiceMemo/parakeet-fp32/nvidia');SR=16000;SECONDS=30

def load_wav(p):
 with wave.open(str(p)) as f:return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768

class Fp16SafeLayerNorm(torch.nn.Module):
 """LayerNorm rewritten so every intermediate stays inside the FP16 range.

 Parakeet's residual stream has outlier channels up to ~1.2e4, so (x - mean)^2 reaches ~1.5e8
 and its row sum ~1.5e9: far beyond FP16's 65504 when a GPU stores intermediates in FP16
 (LiteRT F32_F16). Normalising each row by its own max |x - mean| is algebraically exact:
   d = x - mean(x);  m = max(max|d|, 1e-3);  d' = d / m          (|d'| <= 1)
   LN(x) = gamma * d' / sqrt(mean(d'^2) + (sqrt(eps) / m)^2) + beta
 The 1e-3 floor only matters for (near-)constant rows, where LN output is ~0 either way.
 """
 def __init__(self,ln):
  super().__init__();self.weight=ln.weight;self.bias=ln.bias;self.root_eps=float(ln.eps)**0.5
 def forward(self,x):
  d=x-x.mean(-1,keepdim=True)
  m=torch.clamp(d.abs().amax(-1,keepdim=True),min=1e-3)
  dn=d/m
  return dn*torch.rsqrt((dn*dn).mean(-1,keepdim=True)+(self.root_eps/m)**2)*self.weight+self.bias

def make_layernorms_fp16_safe(model):
 n=0
 for parent in model.encoder.modules():
  for name,child in list(parent.named_children()):
   if isinstance(child,torch.nn.LayerNorm):setattr(parent,name,Fp16SafeLayerNorm(child));n+=1
 return n

class Encoder(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,features,mask):
  return self.model.get_audio_features(input_features=features,attention_mask=mask).pooler_output

class DecodeStep(torch.nn.Module):
 def __init__(self,model):
  super().__init__();self.decoder=model.decoder;self.joint=model.joint
 def forward(self,token,h,c,frame):
  out,(h2,c2)=self.decoder.lstm(self.decoder.embedding(token),(h,c))
  dec=self.decoder.decoder_projector(out)[:,0]
  return self.joint(encoder_hidden_states=frame,decoder_hidden_states=dec),h2,c2

class Predict(torch.nn.Module):
 """Prediction network only: run after a non-blank emission (or once at the start)."""
 def __init__(self,model):super().__init__();self.decoder=model.decoder
 def forward(self,token,h,c):
  out,(h2,c2)=self.decoder.lstm(self.decoder.embedding(token),(h,c))
  return self.decoder.decoder_projector(out)[:,0],h2,c2

class Joint(torch.nn.Module):
 """Joint network for one encoder frame with argmax in-graph: returns (token id, duration index)."""
 def __init__(self,model):super().__init__();self.joint=model.joint;self.V=model.config.vocab_size
 def forward(self,frame,dec):
  lg=self.joint(encoder_hidden_states=frame,decoder_hidden_states=dec)
  return torch.argmax(lg[:,:self.V],dim=-1).to(torch.int32),torch.argmax(lg[:,self.V:],dim=-1).to(torch.int32)

def check_split_parity(model,proc):
 """Predict+Joint reproduce DecodeStep token-for-token on the English clips."""
 enc=Encoder(model).eval();pr=Predict(model).eval();jo=Joint(model).eval();cfg=model.config;ok=True
 for sid in ('english-dialogue-short','english-dialogue-proper-names','english-dialogue-mixed-fragments'):
  audio=load_wav(ROOT/'benchmark-inputs/article-samples'/sid/'audio.wav')[:SECONDS*SR]
  x=proc(audio,sampling_rate=SR,return_tensors='pt')
  with torch.inference_mode():
   e=enc(x['input_features'],x['attention_mask'])
   valid=int(model.encoder._get_subsampling_output_length(x['attention_mask'].sum(-1))[0])
   ref=decode(DecodeStep(model).eval(),cfg,e,valid)
   h=torch.zeros(2,1,640);c=torch.zeros(2,1,640);dec,h2,c2=pr(torch.tensor([[cfg.blank_token_id]],dtype=torch.int32),h,c)
   t=0;sym=0;ids=[]
   while t<valid:
    k,du=jo(e[:,t],dec);k=int(k);du=cfg.durations[int(du)]
    if k!=cfg.blank_token_id:
     ids.append(k);h,c=h2,c2;dec,h2,c2=pr(torch.tensor([[k]],dtype=torch.int32),h,c);sym+=1
     if du==0 and sym>=10:du=1
    elif du==0:du=1
    if du>0:sym=0
    t+=du
  same=ids==ref;ok&=same;print('split parity',sid,'tokens',len(ids),'identical' if same else 'DIFFERENT',flush=True)
 return ok

def check_parity(model,proc):
 """Our wrappers reproduce transformers' generate() token-for-token on the English clips (whole)."""
 enc=Encoder(model).eval();step=DecodeStep(model).eval();cfg=model.config;ok=True
 for sid in ('english-dialogue-short','english-dialogue-proper-names','english-dialogue-mixed-fragments'):
  audio=load_wav(ROOT/'benchmark-inputs/article-samples'/sid/'audio.wav')[:SECONDS*SR]
  x=proc(audio,sampling_rate=SR,return_tensors='pt')
  with torch.inference_mode():
   ref=model.generate(**x,return_dict_in_generate=True,do_sample=False,num_beams=1,max_new_tokens=8192)
   ref_ids=[t for t in ref.sequences[0].tolist() if t!=cfg.blank_token_id]
   e=enc(x['input_features'],x['attention_mask'])
   valid=int(model.encoder._get_subsampling_output_length(x['attention_mask'].sum(-1))[0])
   ids=decode(step,cfg,e,valid)
  same=ids==ref_ids;ok&=same;print('parity',sid,'tokens',len(ids),'identical' if same else 'DIFFERENT',flush=True)
 return ok

@torch.inference_mode()
def decode(step,cfg,enc_out,valid,max_symbols=10):
 blank=cfg.blank_token_id;V=cfg.vocab_size;D=cfg.durations
 h=torch.zeros(2,1,640);c=torch.zeros(2,1,640);last=blank;t=0;sym=0;ids=[]
 # decoder state/output for the current label context; recomputed only after a non-blank emission
 while t<valid:
  logits,h2,c2=step(torch.tensor([[last]],dtype=torch.int32),h,c,enc_out[:,t])
  tok=int(logits[0,:V].argmax());du=D[int(logits[0,V:].argmax())]
  if tok!=blank:
   ids.append(tok);last=tok;h,c=h2,c2;sym+=1
   if du==0 and sym>=max_symbols:du=1
  else:
   if du==0:du=1
  if du>0:sym=0
  t+=du
 return ids

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');ap.add_argument('--quant',choices=['int8','fp16','none'],default='int8')
 ap.add_argument('--fp16-safe-layernorm',action='store_true',help='rewrite encoder LayerNorms for GPU F32_F16 (FP16 intermediates)')
 ap.add_argument('--only',choices=['encoder','decoder'],help='with --split: write one file only (the encoder conversion needs most of the memory)')
 ap.add_argument('--skip-parity',action='store_true',help='skip the parity checks (already passed for this checkpoint)')
 ap.add_argument('--split',action='store_true',help='write <stem>_encoder.tflite (encode) and <stem>_decoder.tflite (predict, joint; fp32)')
 ap.add_argument('--model-dir',type=pathlib.Path,default=MODEL_DIR,help='Transformers-format Parakeet TDT checkpoint (default: NVIDIA v3)')
 ap.add_argument('--name',default='parakeet_tdt_0.6b_v3',help='output file stem prefix, e.g. parakeet_primeline')
 ap.add_argument('--source-revision',default='7c35754d166cca382ad1e53e68b01e7c575f3a1d',help='recorded in the sidecar JSON')
 ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('/Volumes/X10 Pro/VoiceMemo/parakeet-litert'));a=ap.parse_args()
 torch.set_num_threads(4)
 proc=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
 model=AutoModelForTDT.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').eval()
 if a.fp16_safe_layernorm:print('fp16-safe LayerNorms:',make_layernorms_fp16_safe(model),flush=True)
 if not a.skip_parity and not check_parity(model,proc):sys.exit('Wrapper parity failed; not exporting.')
 if a.split and not a.skip_parity and not check_split_parity(model,proc):sys.exit('Split decoder parity failed; not exporting.')
 if a.check_only:return
 import litert_torch
 from litert_torch.generative.quantize import quant_recipes
 x=proc(np.zeros(SECONDS*SR,np.float32),sampling_rate=SR,return_tensors='pt')
 feats=x['input_features'];mask=x['attention_mask'].to(torch.int32)
 print('feature shape',tuple(feats.shape),flush=True)
 step_args=(torch.zeros(1,1,dtype=torch.int32),torch.zeros(2,1,640),torch.zeros(2,1,640),torch.zeros(1,640))
 qc={'int8':quant_recipes.full_dynamic_recipe(),'fp16':quant_recipes.full_fp16_recipe(),'none':None}[a.quant]
 t=time.perf_counter()
 a.out.mkdir(parents=True,exist_ok=True)
 if a.split:
  stem=f'{a.name}_{SECONDS}s_{a.quant}'+('_lnsafe' if a.fp16_safe_layernorm else '')
  dec_path=a.out/f'{stem}_decoder.tflite'
  if a.only!='encoder':
   dconv=litert_torch.signature('predict',Predict(model).eval(),(torch.zeros(1,1,dtype=torch.int32),torch.zeros(2,1,640),torch.zeros(2,1,640)))
   dconv=dconv.signature('joint',Joint(model).eval(),(torch.zeros(1,640),torch.zeros(1,640)))
   dconv.convert().export(str(dec_path));print('exported',dec_path,dec_path.stat().st_size,'bytes',flush=True)
  if a.only!='decoder':
   enc_path=a.out/f'{stem}_encoder.tflite'
   econv=litert_torch.signature('encode',Encoder(model).eval(),(feats,mask))
   (econv.convert(quant_config=qc) if qc else econv.convert()).export(str(enc_path))
   print('exported',enc_path,enc_path.stat().st_size,'bytes in',round(time.perf_counter()-t),'s',flush=True)
  (a.out/f'{stem}_split.json').write_text(json.dumps({'seconds':SECONDS,'feature_frames':int(feats.shape[1]),'encoder_quant':a.quant,'decoder_quant':'none','source_revision':a.source_revision},indent=1)+'\n')
  return
 conv=litert_torch.signature('encode',Encoder(model).eval(),(feats,mask)).signature('decode_step',DecodeStep(model).eval(),step_args)
 edge=conv.convert(quant_config=qc) if qc else conv.convert()
 path=a.out/f'{a.name}_{SECONDS}s_{a.quant}_step.tflite'
 edge.export(str(path));print('exported',path,path.stat().st_size,'bytes in',round(time.perf_counter()-t),'s',flush=True)
 (a.out/f'{path.stem}.json').write_text(json.dumps({'seconds':SECONDS,'feature_frames':int(feats.shape[1]),'quant':a.quant,'source_revision':a.source_revision},indent=1)+'\n')
if __name__=='__main__':main()
