"""CPU teaching experiment. Synthetic data; NOT a VLA or Kaiwu benchmark."""
from pathlib import Path
import argparse, csv, hashlib, json, time
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(train_size=256, test_size=256, epochs=100, learning_rate=0.035,
              batch_size=32, random_candidates=16, bm_candidates=4,
              bits_per_dimension=3, action_dimensions=2, tau_error=0.5,
              l2=0.0001, seeds=[7, 19, 42])

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=-1, keepdims=True)

def states():
    # Enumerate all 2^6 binary states. Exact enumeration replaces Kaiwu SA.
    h = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)
    u = (h.reshape(64, 2, 3) @ np.array([4., 2., 1.]) + .5) / 8
    z = np.vectorize(NormalDist().inv_cdf)(u)
    return h, z

H, Z = states()
PAIR = H[:, [i for i in range(6) for j in range(i+1, 6)]] * H[:, [j for i in range(6) for j in range(i+1, 6)]]

def features(context):
    # E(c,h) = linear(h) + pairwise(h,h) + conditional(c,h).
    # Small fixed context encoder replaces the video's learned Transformer.
    b = len(context)
    return np.concatenate([np.broadcast_to(H, (b,64,6)),
                           np.broadcast_to(PAIR, (b,64,15)),
                           (context[:,None,:,None] * H[None,:,None,:]).reshape(b,64,12)], axis=-1)

def frozen_generator(context, noise):
    """No trainable weights. This analytic mapping stands in for frozen VLA."""
    return context[..., :2] + .35 * noise

def dataset(rng, n):
    c = rng.uniform(-1,1,(n,2))
    # Invented teacher law; no measured robot demonstrations.
    desired_noise = np.stack([.8*c[:,0]-.5*c[:,1], .6*c[:,0]+.7*c[:,1]], axis=-1)
    target = frozen_generator(c, desired_noise)
    return c, target

def errors(c, target):
    pred = frozen_generator(c[:,None,:], Z[None,:,:])
    return ((pred-target[:,None,:])**2).mean(axis=-1)

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def svg_lines(path, series, title, ylabel):
    colors=['#2563eb','#e87924','#16a085']; W,Hh=900,420; left,top,pw,ph=85,55,760,290
    values=[v for _,ys in series for v in ys]; lo=0; hi=max(values)*1.12 or 1
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}">', '<rect width="100%" height="100%" fill="white"/>',f'<text x="40" y="28" font-family="sans-serif" font-size="19">{title}</text>']
    for k in range(5):
        y=top+ph-k*ph/4; val=hi*k/4
        out += [f'<path d="M{left},{y}h{pw}" stroke="#e2e8f0"/>',f'<text x="12" y="{y+5}" font-size="13">{val:.4f}</text>']
    for idx,(name,ys) in enumerate(series):
        pts=' '.join(f'{left+i*pw/max(1,len(ys)-1):.2f},{top+ph-v/hi*ph:.2f}' for i,v in enumerate(ys))
        out += [f'<polyline points="{pts}" fill="none" stroke="{colors[idx]}" stroke-width="2.5"/>',f'<text x="{left+idx*250}" y="390" fill="{colors[idx]}" font-size="13">{name}</text>']
    out += [f'<text x="400" y="365" font-size="13">Epoch (0 to {len(series[0][1])-1})</text>',f'<text x="85" y="48" font-size="12">{ylabel}</text>','</svg>']
    path.write_text('\n'.join(out))

def learning_rate_at(step,total_steps,warmup_steps,peak,min_ratio):
    if warmup_steps and step <= warmup_steps:
        return peak*step/warmup_steps
    progress=min(1.,max(0.,(step-warmup_steps)/max(1,total_steps-warmup_steps)))
    return peak*(min_ratio+(1-min_ratio)*.5*(1+np.cos(np.pi*progress)))


def run(seed):
    rng=np.random.default_rng(seed); c,t=dataset(rng,CONFIG['train_size']); tc,tt=dataset(rng,CONFIG['test_size'])
    phi=features(c); err=errors(c,t); tp=features(tc); te=errors(tc,tt)
    theta=np.zeros(phi.shape[-1]); m=np.zeros_like(theta); v=np.zeros_like(theta); step=0; rows=[]
    # Same test contexts and states for all methods; no held-out label enters selection.
    fixed_candidates=rng.integers(0,64,(len(tc),20))
    random_ids=rng.integers(0,64,len(tc)); baseline=float(te[np.arange(len(tc)),random_ids].mean())
    def evaluate():
        energy=tp@theta; prob=softmax(-energy)
        selected=fixed_candidates[np.arange(len(tc)),np.argmin(np.take_along_axis(energy,fixed_candidates,1),1)]
        return float((prob*te).sum(1).mean()),float(te[np.arange(len(tc)),selected].mean())
    batches_per_epoch=len(c)//CONFIG['batch_size']
    total_steps=CONFIG['epochs']*batches_per_epoch
    warmup_steps=CONFIG.get('warmup_epochs',0)*batches_per_epoch
    batch_path=ROOT/'results'/f'batches_{seed}.csv'
    batch_path.write_text('')
    start=time.perf_counter()
    for epoch in range(CONFIG['epochs']+1):
        losses=[]; grad_norms=[]; update_norms=[]; kls=[]; entropies=[]; regularizations=[]
        if epoch:
            for ids in np.array_split(rng.permutation(len(c)),len(c)//CONFIG['batch_size']):
                x=phi[ids]; e=err[ids]; prob=softmax(-(x@theta))
                # Four current-model samples + sixteen random states, as in video.
                bm=np.stack([rng.choice(64,4,p=p) for p in prob]); random=rng.integers(0,64,(len(ids),16))
                cand=np.concatenate([bm,random],1); sub=x[np.arange(len(ids))[:,None],cand]; ce=np.take_along_axis(e,cand,1)
                norm=(ce-ce.mean(1,keepdims=True))/(ce.std(1,keepdims=True)+1e-6)
                teacher=softmax(-norm/CONFIG['tau_error']); candidate_energy=sub@theta; predicted=softmax(-candidate_energy)
                regularization=CONFIG['l2']*(theta@theta)/2
                loss=-(teacher*np.log(predicted+1e-12)).sum(1).mean()+regularization
                entropy=-(teacher*np.log(teacher+1e-12)).sum(1).mean()
                kl=(teacher*(np.log(teacher+1e-12)-np.log(predicted+1e-12))).sum(1).mean()
                entropies.append(float(entropy)); kls.append(float(kl))
                regularizations.append(float(regularization))
                # Gradient is through energy only. No differentiation of generator or sampling.
                grad=np.einsum('bk,bkf->f',teacher-predicted,sub)/len(ids)+CONFIG['l2']*theta
                step+=1; m=.9*m+.1*grad; v=.999*v+.001*grad**2
                lr=learning_rate_at(step,total_steps,warmup_steps,CONFIG['learning_rate'],CONFIG.get('min_lr_ratio',.1)) if CONFIG.get('lr_schedule','constant')=='cosine' else CONFIG['learning_rate']
                delta=lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
                grad_norms.append(float(np.linalg.norm(grad)))
                update_norms.append(float(np.linalg.norm(delta)))
                theta-=delta
                losses.append(float(loss))
                if step % CONFIG.get('batch_log_every',50)==0 or step==1 or step==total_steps:
                    batch_row={'optimizer_step':step,'batch/epoch':epoch,'batch/loss':float(loss),
                               'batch/lr':lr,'batch/grad_norm':float(np.linalg.norm(grad)),
                               'batch/energy_mean':float(candidate_energy.mean()),
                               'batch/energy_std':float(candidate_energy.std(axis=1).mean()),
                               'batch/kl_target_pred':float(kl)}
                    empty=batch_path.stat().st_size==0
                    with batch_path.open('a',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(batch_row))
                        if empty:w.writeheader()
                        w.writerow(batch_row)
        expected,selected=evaluate()
        if epoch % CONFIG.get('log_every',10) == 0 or epoch == CONFIG['epochs']:
            print(f'seed={seed} epoch={epoch}/{CONFIG["epochs"]} test_mse={expected:.6f}',flush=True)
        eval_energy=tp@theta
        fixed_energy=eval_energy[np.arange(len(tc))[:,None],fixed_candidates]
        row=dict(epoch=epoch,loss=float(np.mean(losses)) if losses else '',
                 heldout_expected_mse=expected,heldout_selected_mse=selected,random_baseline_mse=baseline,
                 **{'train/grad_norm':float(np.mean(grad_norms)) if grad_norms else '',
                    'train/update_norm':float(np.mean(update_norms)) if update_norms else '',
                    'train/lr':lr if epoch else (0. if warmup_steps and CONFIG.get('lr_schedule')=='cosine' else CONFIG['learning_rate']),
                    'energy/all_states_mean':float(eval_energy.mean()),
                    'energy/within_context_std':float(eval_energy.std(axis=1).mean()),
                    'energy/bm_expected':float((softmax(-eval_energy)*eval_energy).sum(axis=1).mean()),
                    'energy/selected_20_mean':float(fixed_energy.min(axis=1).mean()),
                    'energy/selected_gap':float((fixed_energy.mean(axis=1)-fixed_energy.min(axis=1)).mean()),
                    'train/parameter_norm':float(np.linalg.norm(theta)),
                    'train/target_entropy':float(np.mean(entropies)) if entropies else '',
                    'train/kl_target_pred':float(np.mean(kls)) if kls else '',
                    'train/regularization':float(np.mean(regularizations)) if regularizations else '',
                    'train/optimizer_step':step,
                    'time/elapsed_seconds':time.perf_counter()-start,
                    'eval/uniform_expected_mse':float(te.mean())})
        rows.append(row)
        # Persist each epoch so logs survive interruption; no network upload here.
        with (ROOT/'results'/f'training_{seed}.csv').open('w' if epoch==0 else 'a',newline='') as log_file:
            writer=csv.DictWriter(log_file,fieldnames=list(row))
            if epoch==0: writer.writeheader()
            writer.writerow(row)
        if epoch > 0 and epoch % CONFIG.get('checkpoint_every',500) == 0:
            checkpoint_dir=ROOT/'results'/'checkpoints'
            checkpoint_dir.mkdir(parents=True,exist_ok=True)
            checkpoint_path=checkpoint_dir/f'seed_{seed}_epoch_{epoch:05d}.npz'
            np.savez(checkpoint_path,theta=theta,adam_m=m,adam_v=v,
                     optimizer_step=step,epoch=epoch,seed=seed,
                     config_json=json.dumps(CONFIG),rng_state_json=json.dumps(rng.bit_generator.state))
            write_csv(ROOT/'results'/f'training_{seed}.csv',rows)
            print(f'Checkpoint saved: {checkpoint_path}',flush=True)
    np.savez(ROOT/'results'/f'checkpoint_{seed}.npz',theta=theta,seed=seed)
    np.savez(ROOT/'data'/f'synthetic_{seed}.npz',train_context=c,train_target=t,test_context=tc,test_target=tt)
    write_csv(ROOT/'results'/f'training_{seed}.csv',rows)
    energy=tp@theta; chosen=fixed_candidates[np.arange(len(tc)),np.argmin(np.take_along_axis(energy,fixed_candidates,1),1)]
    preview=[]
    for i in range(12):
        action=frozen_generator(tc[i],Z[chosen[i]])
        preview.append(dict(example=i,context=tc[i].tolist(),target=tt[i].tolist(),selected_bits=H[chosen[i]].tolist(),noise=Z[chosen[i]].tolist(),predicted_action=action.tolist(),mse=float(te[i,chosen[i]])))
    (ROOT/'results'/f'inference_{seed}.json').write_text(json.dumps(preview,indent=2))
    result=dict(seed=seed,random_one_mse=baseline,untrained_bm_expected_mse=rows[0]['heldout_expected_mse'],
                trained_bm_expected_mse=rows[-1]['heldout_expected_mse'],trained_best_of_20_mse=rows[-1]['heldout_selected_mse'],
                oracle_all_64_mse=float(te.min(1).mean()),seconds=time.perf_counter()-start)
    return result,rows

def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--train-size',type=int,default=256)
    parser.add_argument('--test-size',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,help='Run only this random seed')
    parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--checkpoint-every',type=int,default=500)
    parser.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant')
    parser.add_argument('--warmup-epochs',type=int,default=0)
    parser.add_argument('--min-lr-ratio',type=float,default=.1)
    parser.add_argument('--batch-log-every',type=int,default=50)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.train_size < 32 or args.test_size < 12 or args.epochs < 1:
        parser.error('train-size >=32, test-size >=12, epochs >=1 required')
    if args.checkpoint_every < 1:
        parser.error('checkpoint-every must be positive')
    if args.log_every < 1:
        parser.error('log-every must be positive')
    if not 0 <= args.warmup_epochs < args.epochs or not 0 <= args.min_lr_ratio <= 1 or args.batch_log_every < 1:
        parser.error('Require 0 <= warmup-epochs < epochs, min-lr-ratio in [0,1], batch-log-every >=1')
    if args.warmup_epochs and args.lr_schedule=='constant':
        parser.error('warmup requires --lr-schedule cosine')
    CONFIG.update(lr_schedule=args.lr_schedule,warmup_epochs=args.warmup_epochs,min_lr_ratio=args.min_lr_ratio,batch_log_every=args.batch_log_every)
    CONFIG.update(train_size=args.train_size,test_size=args.test_size,epochs=args.epochs,log_every=args.log_every,checkpoint_every=args.checkpoint_every)
    if args.seed is not None:
        CONFIG['seeds']=[args.seed]
    if args.output_dir:
        ROOT=args.output_dir.resolve()
        ROOT.mkdir(parents=True,exist_ok=True)
    print(f"Synthetic teaching experiment: {CONFIG['train_size']} train, {CONFIG['test_size']} test, {CONFIG['epochs']} epochs per seed",flush=True)
    if args.smoke: CONFIG.update(epochs=5,seeds=[7])
    for folder in ['results','data']: (ROOT/folder).mkdir(exist_ok=True)
    (ROOT/'results'/'config.json').write_text(json.dumps(CONFIG,indent=2))
    results=[]; curves=[]
    for seed in CONFIG['seeds']:
        result,rows=run(seed); results.append(result);curves.append(rows); print(json.dumps(result))
    (ROOT/'results'/'config.json').write_text(json.dumps(CONFIG,indent=2))
    (ROOT/'results'/'summary.json').write_text(json.dumps(results,indent=2))
    write_csv(ROOT/'results'/'summary.csv',results)
    svg_lines(ROOT/'results'/'training_curve.svg',[(label,[float(np.mean([c[i][key] for c in curves])) for i in range(len(curves[0]))]) for label,key in [('Random one (baseline)','random_baseline_mse'),('Trained BM: expected one','heldout_expected_mse'),('Trained energy: best of 20','heldout_selected_mse')]],f'Synthetic held-out action error ({len(CONFIG["seeds"])} seeds)','Mean squared error')
    svg_lines(ROOT/'results'/'loss_curve.svg',[('SoftNCE + regularization',[float(np.mean([c[i]['loss'] for c in curves])) for i in range(1,len(curves[0]))])],'Actual training loss','Loss')
    import shutil
    shutil.copyfile(__file__, ROOT/'results'/'train_source.py')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in {'.py','.npz','.json','.csv'}}
    (ROOT/'results'/'sha256.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
