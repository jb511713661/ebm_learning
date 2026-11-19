"""Upload measured CSVs only. Authenticate using `swanlab login` first."""
import argparse, csv, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workspace',default=None); p.add_argument('--project',default='ebm-learning-lab'); p.add_argument('--run-dir',type=Path); args=p.parse_args()
    import swanlab
    root=Path(__file__).resolve().parents[1]
    if args.run_dir: root=args.run_dir.resolve()
    config=json.loads((root/'results/config.json').read_text())
    for seed in config['seeds']:
        swanlab.init(project=args.project,workspace=args.workspace,
                     experiment_name=f'synthetic-ebm-n{config["train_size"]}-e{config["epochs"]}-seed-{seed}',
                     config={**config,'seed':seed,'scope':'SYNTHETIC TEACHING ONLY; no VLA/Kaiwu benchmark'})
        with (root/'results'/f'training_{seed}.csv').open() as f:
            for row in csv.DictReader(f):
                names={'loss':'train/loss','heldout_expected_mse':'eval/bm_expected_mse',
                       'heldout_selected_mse':'eval/selected_20_mse',
                       'random_baseline_mse':'eval/random_one_mse'}
                metrics={names.get(k,k):float(v) for k,v in row.items() if k!='epoch' and v!=''}
                swanlab.log(metrics,step=int(row['epoch']))
        swanlab.finish()
        batch_file=root/'results'/f'batches_{seed}.csv'
        if batch_file.exists() and batch_file.stat().st_size:
            swanlab.init(project=args.project,workspace=args.workspace,
                         experiment_name=f'synthetic-ebm-batches-e{config["epochs"]}-seed-{seed}',
                         config={**config,'seed':seed,'x_axis':'optimizer updates','scope':'SYNTHETIC TEACHING ONLY'})
            with batch_file.open() as f:
                for row in csv.DictReader(f):
                    swanlab.log({k:float(v) for k,v in row.items() if k!='optimizer_step'},step=int(row['optimizer_step']))
            swanlab.finish()
    print('Uploaded actual logged metrics. Open SwanLab to inspect/export charts.')

if __name__=='__main__': main()
