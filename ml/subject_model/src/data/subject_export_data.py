"""Prepare subject-run training calibration and validation only; never read test tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from src.models.tinyeegnet import TinyEEGNet
from src.train import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    # Read only the preselection data file, which contains no test data.
    data=torch.load(args.run/'development_tensors.pt',weights_only=True)
    manifest=json.loads((args.run/'split_manifest.json').read_text())
    assert all(t['session']=='0train' and t['subject']==1 for n in ('train','validation') for t in manifest[n]['trials'])
    args.out.mkdir(parents=True,exist_ok=False)
    old={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ('artifacts','results') for p in Path(root).rglob('*') if p.is_file() and not p.is_relative_to(args.out)}
    save_json(args.out/'preserved_hashes.json',old)
    torch.set_num_threads(4)
    model=TinyEEGNet(dropout=.5).eval()
    model.load_state_dict(torch.load(args.run/'best_checkpoint.pt',weights_only=True)['model_state'])
    with torch.no_grad(): logits=torch.cat([model(x[None]) for x in data['validation']['x']])
    torch.save({'calibration':data['train']['x'],'calibration_y':data['train']['y'],
                'development_x':data['validation']['x'],'development_y':data['validation']['y'],
                'reference_logits':logits},args.out/'export_data.pt')
    shutil.copyfile(args.run/'preprocess_params.json',args.out/'float_preprocess_params.json')
    save_json(args.out/'data_manifest.json',{'seed':42,'calibration_trials':manifest['train']['trials'],
               'development_trials':manifest['validation']['trials'],'calibration_count':len(data['train']['y']),
               'calibration_class_counts':torch.bincount(data['train']['y']).tolist(),
               'checkpoint':str(args.run/'best_checkpoint.pt'),'held_out_test_used':False})

if __name__=='__main__': main()
