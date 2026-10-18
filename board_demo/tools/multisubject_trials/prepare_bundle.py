#!/usr/bin/env python3
"""Verify/acquire the unfused v4 bundle and pinned sources without modifying a model.

Reads an existing local Git repository. Downloads occur only with --download.
Private GitHub access may use GITHUB_TOKEN from the environment; tokens are never
written or printed. EEG datasets are never downloaded by this utility.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, urllib.request

REVISION='ad61c79de17243f5befce069a1e9ca458057f43e'
EDGE_REVISION='c616ea07e42926c6490d43a220f4bc0915b8f54f'
REFERENCE_SHA='d41cf8bca4149071b737cbbf8ad73f49e1f1223a2c6ef3b3a88108ec7aa66a70'
ROOT=Path(__file__).resolve().parents[2]
def sha(b): return hashlib.sha256(b).hexdigest()
def put(path,data):
    if path.exists():
        if path.read_bytes()!=data: raise ValueError(f'Refusing changed-file overwrite: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def git(repo,*args):
    result=subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    return result.stdout
class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        result=super().redirect_request(request,fp,code,msg,headers,newurl)
        if result is not None:result.remove_header('Authorization')
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo',type=Path,default=ROOT)
    p.add_argument('--edge-repo',type=Path,help='Existing Edge_Deploy checkout containing the pinned reference commit')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/multisubject-v4')
    p.add_argument('--download',action='store_true')
    args=p.parse_args();manifest=json.loads(Path(__file__).with_name('release-assets.json').read_text())
    out=args.out.resolve();bundle=out/'bundle';upstream=out/'upstream'
    token=os.environ.get('GITHUB_TOKEN');headers={'Accept':'application/vnd.github+json','User-Agent':'EdgeBCI-artifact-verifier'}
    if token:headers['Authorization']='Bearer '+token
    opener=urllib.request.build_opener(SafeRedirect())
    missing=[a for a in manifest['assets'] if not (bundle/a['name']).is_file()]
    release=None
    if missing:
        if not args.download: raise FileNotFoundError('Missing release assets; supply them locally or explicitly use --download')
        request=urllib.request.Request('https://api.github.com/repos/ArmExecuTorchEEG/Reproduce_Baselines/releases/tags/v0.5-distill',headers=headers)
        with opener.open(request,timeout=60) as response:release=json.load(response)
    for asset in manifest['assets']:
        path=bundle/asset['name']
        if path.is_file():data=path.read_bytes()
        else:
            matches=[a for a in release['assets'] if a['name']==asset['name']]
            if len(matches)!=1:raise ValueError('Release asset missing/ambiguous: '+asset['name'])
            h={**headers,'Accept':'application/octet-stream'}
            with opener.open(urllib.request.Request(matches[0]['url'],headers=h),timeout=120) as response:data=response.read()
        if len(data)!=asset['bytes'] or sha(data)!=asset['sha256']:raise ValueError('Release checksum mismatch: '+asset['name'])
        put(path,data)
    put(bundle/'release-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    paths=git(args.source_repo,'ls-tree','-r','--name-only',REVISION,'src','configs').decode().splitlines()
    if not paths:raise ValueError('Pinned source snapshot missing')
    for name in paths:put(upstream/name,git(args.source_repo,'show',f'{REVISION}:{name}'))
    ref=bundle/'offline_int8_predictions.json'
    if not ref.is_file():
        if args.edge_repo is None:raise FileNotFoundError('Supply --edge-repo or the original saved desktop reference')
        data=git(args.edge_repo,'show',f'{EDGE_REVISION}:host/reference/offline_int8_predictions.json')
        if sha(data)!=REFERENCE_SHA:raise ValueError('Saved desktop reference checksum mismatch')
        put(ref,data)
    if sha(ref.read_bytes())!=REFERENCE_SHA:raise ValueError('Saved desktop reference checksum mismatch')
    print(f'PASS: {len(manifest["assets"])} release assets, {len(paths)} pinned source files, saved1080-trial reference; no training or hardware access')
if __name__=='__main__':main()
