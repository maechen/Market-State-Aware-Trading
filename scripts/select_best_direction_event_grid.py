from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

base=Path('data/transformer_npy/event_grid')
rows=[]
for fm in sorted(base.glob('k*_h*/fold_metrics.csv')):
    df=pd.read_csv(fm)
    run_dir=fm.parent
    k=float(run_dir.name.split('_')[0][1:])
    h=int(run_dir.name.split('_')[1][1:])
    def wavg(v,w):
        return float(np.average(v,weights=w)) if np.sum(w)>0 else float('nan')
    row={
      'run_dir':str(run_dir),'k':k,'horizon':h,
      'val_coverage_weighted':wavg(df['val_dir_coverage'], df['val_pred_rows']),
      'val_event_count_total':float(df['val_dir_event_count'].sum()),
      'val_dir_mae_weighted':wavg(df['val_dir_mae_event'], df['val_dir_event_count']),
      'val_dir_acc_weighted':wavg(df['val_dir_acc_event'], df['val_dir_event_count']),
      'val_majority_acc_weighted':wavg(df['val_dir_majority_acc_event'], df['val_dir_event_count']),
      'val_acc_minus_majority_weighted':wavg(df['val_dir_acc_minus_majority_event'], df['val_dir_event_count']),
      'test_coverage_weighted':wavg(df['test_dir_coverage'], df['test_pred_rows']),
      'test_event_count_total':float(df['test_dir_event_count'].sum()),
      'test_dir_mae_weighted':wavg(df['test_dir_mae_event'], df['test_dir_event_count']),
      'test_dir_acc_weighted':wavg(df['test_dir_acc_event'], df['test_dir_event_count']),
      'test_majority_acc_weighted':wavg(df['test_dir_majority_acc_event'], df['test_dir_event_count']),
      'test_acc_minus_majority_weighted':wavg(df['test_dir_acc_minus_majority_event'], df['test_dir_event_count']),
      'test_reg_acc_weighted':wavg(df['test_reg_acc'], df['test_num_samples']),
      'val_mae_std': float(df['val_dir_mae_event'].std(ddof=0)),
      'val_up_rate': wavg(df['val_dir_up_rate_event'], df['val_dir_event_count']),
      'test_up_rate': wavg(df['test_dir_up_rate_event'], df['test_dir_event_count']),
    }
    row['eligible']= bool(row['val_coverage_weighted']>=0.40 and row['val_acc_minus_majority_weighted']>0 and 0.35<=row['val_up_rate']<=0.65 and 0.35<=row['test_up_rate']<=0.65)
    rows.append(row)
out=pd.DataFrame(rows)
out['rank']=np.nan
eligible=out[out['eligible']].copy().sort_values(['val_dir_mae_weighted','val_acc_minus_majority_weighted','val_coverage_weighted','val_mae_std'], ascending=[True,False,False,True])
out.loc[eligible.index,'rank']=np.arange(1,len(eligible)+1)
out=out.drop(columns=['val_mae_std','val_up_rate','test_up_rate'])
out.to_csv(base/'selection_summary.csv', index=False)
print(out.sort_values(['rank','val_dir_mae_weighted'], na_position='last').to_string(index=False))
