"""One predeclared Participant-1 run; session 2 is loaded only after selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import yaml

from src.data.preprocess import Split, LABELS
from src.experiments.dropout_cv import load_allowed
from src.models.tinyeegnet import TinyEEGNet
from src.train import train, seed_everything, save_json, export_check


def partition(x, y, trials, seed=42):
    ti, vi = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    mean = x[ti].mean(axis=(0, 3), keepdims=True)
    std = x[ti].std(axis=(0, 3), keepdims=True)
    assert np.isfinite(std).all() and (std > 0).all()
    def make(indices):
        return Split(torch.from_numpy(((x[indices]-mean)/std).astype(np.float32)),
                     torch.tensor(y[indices], dtype=torch.int64), tuple(trials[i] for i in indices))
    groups = {'train': make(ti), 'validation': make(vi)}
    assert not {t.id for t in groups['train'].trials} & {t.id for t in groups['validation'].trials}
    ordered = sorted(trials, key=lambda t: (t.run, t.cue_sample))
    assert all(a.run != b.run or a.stop <= b.cue_sample-500 for a,b in zip(ordered, ordered[1:]))
    return groups, mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/subject1/baseline.yaml'))
    parser.add_argument('--run-name', default='subject1-seed42-20261016')
    args = parser.parse_args()
    assert Path(args.run_name).name == args.run_name
    config = yaml.safe_load(args.config.read_text())
    original = yaml.safe_load(Path('configs/default.yaml').read_text())
    assert {k:v for k,v in config.items() if k!='protocol'} == {k:v for k,v in original.items() if k!='protocol'}
    out = Path('artifacts')/args.run_name
    results = Path('results')/args.run_name
    old = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ('artifacts','results') for p in Path(root).rglob('*') if p.is_file()}
    out.mkdir(exist_ok=False); results.mkdir(exist_ok=False)
    save_json(out/'preserved_hashes.json',old)
    seed_everything(config['seed'], config['training']['threads'])
    started=time.monotonic()
    (x,y,trials), sources, audit = load_allowed([(1,'0train')])
    assert x.shape == (144,1,22,500) and np.bincount(y).tolist()==[72,72]
    groups,mean,std=partition(x,y,trials,config['seed'])
    params=json.loads(Path('artifacts/combined-seed42-20261016/preprocess_params.json').read_text())
    params.update(mean=mean.reshape(22).tolist(),std=std.reshape(22).tolist(),quantization=None)
    save_json(out/'preprocess_params.json',params)
    save_json(out/'label_mapping.json',LABELS); save_json(out/'config.json',config)
    torch.save({n:{'x':g.x,'y':g.y} for n,g in groups.items()},out/'development_tensors.pt')
    model=TinyEEGNet(dropout=.5)
    training_start=time.monotonic()
    history=train(model,groups['train'],groups['validation'],config,out)
    training_seconds=time.monotonic()-training_start
    checkpoint=torch.load(out/'best_checkpoint.pt',weights_only=True)
    model.load_state_dict(checkpoint['model_state']); model.eval()
    reloaded=TinyEEGNet(dropout=.5).eval(); reloaded.load_state_dict(checkpoint['model_state'])
    with torch.no_grad():
        torch.testing.assert_close(model(groups['validation'].x),reloaded(groups['validation'].x),rtol=0,atol=0)
        assert model(torch.zeros(1,1,22,500)).shape==(1,2)
    export=export_check(model,out/'tinyeegnet_float.pt2')
    # First access to Session 2: checkpoint selection and all fitting are complete.
    (tx,ty,tt), test_sources,test_audit=load_allowed([(1,'1test')])
    assert tx.shape==(144,1,22,500) and np.bincount(ty).tolist()==[72,72]
    groups['test']=Split(torch.from_numpy(((tx-mean)/std).astype(np.float32)),torch.tensor(ty,dtype=torch.int64),tt)
    assert not {t.id for t in tt} & {t.id for t in trials}
    final={}; predictions={}
    # One final inference pass per split; artifact subsets reuse its predictions.
    for name,g in groups.items():
        with torch.no_grad(): logits=model(g.x)
        predictions[name]={'logits':logits,'y':g.y}
        from src.export import metrics
        final[name]=metrics(logits,g.y)
        final[name]['loss']=float(torch.nn.functional.cross_entropy(logits,g.y))
        final[name]['artifact_groups']={}
        for flag,label in ((False,'clean'),(True,'flagged')):
            mask=torch.tensor([t.artifact==flag for t in g.trials])
            final[name]['artifact_groups'][label]=metrics(logits[mask],g.y[mask]) if mask.any() else {'n':0}
    manifest={name:{'shape':list(g.x.shape),'class_counts':torch.bincount(g.y).tolist(),'trials':[asdict(t) for t in g.trials]} for name,g in groups.items()}
    save_json(out/'split_manifest.json',manifest)
    save_json(out/'source_recordings.json',sources+test_sources)
    torch.save({n:{'x':g.x,'y':g.y} for n,g in groups.items()},out/'prepared_tensors.pt')
    torch.save(predictions,out/'predictions.pt'); torch.save(model.state_dict(),out/'tinyeegnet_float.pt')
    report={'seed':42,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':checkpoint['epoch'],'epochs_run':len(history),'training_seconds':training_seconds,'total_seconds':time.monotonic()-started,'selection':'Participant 1 Session 1 validation loss only; Session 2 accessed after selection','metrics':final,'torch_export':export,'reload_exact_match':True,'artifact_audit':audit|test_audit,'checkpoint_sha256':hashlib.sha256((out/'best_checkpoint.pt').read_bytes()).hexdigest(),'versions':{'torch':str(torch.__version__),'numpy':np.__version__},'existing_files_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old.items())}
    save_json(out/'metrics.json',report); save_json(results/'metrics.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Training (dropout active)')
    ax.plot([h['epoch'] for h in history],[h['validation']['loss'] for h in history],label='Validation')
    ax.axvline(checkpoint['epoch'],color='gray',linestyle='--',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='Cross-entropy loss'); ax.legend(); fig.tight_layout()
    fig.savefig(results/'loss_curves.png',dpi=160); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
