"""Compare direct internal-point calibration against the existing 1.28px candidate."""
from _common import parser,run
from pathlib import Path
from datetime import datetime
import numpy as np
from camera_calibration.dataset import read_json,write_json,digest
from camera_calibration.geometry import marker_points
from camera_calibration.solver import solve
from camera_calibration.multipoint import aggregate_internal,fit,evaluate


def compare(source,processed):
    source=Path(source).resolve();processed=Path(processed).resolve()
    previous=read_json(processed/'result.json');original=read_json(processed/'observations.json')
    investigation=source/'diagnostics/residual_investigation'
    edges=read_json(investigation/'edge_observations.json')
    if [(o['sample_id'],o['split']) for o in original]!=[(o['sample_id'],o['split']) for o in edges]:
        raise ValueError('Outer-point comparison membership changed')
    for o in edges:o['points_M']=marker_points().tolist()
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    out=source/'calibration'/f'{stamp}_direct_multipoint';out.mkdir(parents=True)
    internal=[];hashes={}
    for sample in original:
        f=investigation/'internal_detections'/f'{sample["sample_id"]}.json'
        internal.append(aggregate_internal(sample,read_json(f)))
        hashes[str(f.relative_to(source))]=digest(f)
    write_json(out/'internal_observations.json',internal)
    write_json(out/'outer_observations.json',edges)
    train_ids=[o['sample_id'] for o in original if o['split']=='train']
    test_ids=[o['sample_id'] for o in original if o['split']=='test']
    folds=[[train_ids[i] for i in ids] for ids in np.array_split(np.arange(len(train_ids)),5)]
    protocol={'source_capture':str(source),'original_result':str(processed),'source_detection_hashes':hashes,
        'train_ids':train_ids,'test_ids':test_ids,'cv_folds':folds,'marker_side_m':.08,
        'frozen_image_observations':True,'minimum_frames_per_internal_point':6,
        'minimum_internal_points_per_pose':12,'detection':'Cached local cornerSubPix measurements; image-only homography inlier gate. Homography projections are NOT used as optimization observations.',
        'model':'Shared T_W_C and T_E_M, optional shared k1/k2; fixed factory K and 80mm geometry; no per-image fitted pose variables.',
        'loss':'1px component-wise Huber; each pose total cost normalized to four points; all methods use identical loss.',
        'mixed_loss':'Prespecified 50% total weight to outer points and 50% to internal points within each pose; not tuned against held-out metrics.',
        'validation':'Same five contiguous original training folds and original six test poses. Separate common outer-edge and common internal-point metrics; never compare one method internal RMS to another method outer RMS.',
        'initialization':'Each fold starts from r3kit-corner solve on that fold training poses only; full fit initialized from original training poses only.',
        'limits':'This dataset has already been inspected repeatedly; these are diagnostic comparisons, not fresh independent acceptance.',
        'production_changed':False}
    write_json(out/'protocol.json',protocol)
    groups={'outer':{o['sample_id']:o for o in edges},'internal':{o['sample_id']:o for o in internal}}
    mixed={}
    for sid in groups['outer']:
        a=groups['outer'][sid];b=groups['internal'][sid]
        mixed[sid]={**a,'points_M':a['points_M']+b['points_M'],'uv':a['uv']+b['uv'],
                    'point_weights':[.5/len(a['uv'])]*len(a['uv'])+[.5/len(b['uv'])]*len(b['uv'])}
    take=lambda kind,ids:[groups[kind][sid] for sid in ids]
    base={o['sample_id']:o for o in original};cfg=previous['config'];spec=previous['camera']
    seed=solve([base[sid] for sid in train_ids],spec,cfg)
    fold_seeds=[]
    for held in folds:
        fold_seeds.append(solve([base[sid] for sid in train_ids if sid not in held],spec,cfg))
    rows=[]
    try:
        for method,kind,radial in [('edge_radial_baseline','outer',True),('direct_internal_fixed','internal',False),('direct_internal_radial','internal',True),('direct_internal_and_edge_radial','mixed',True)]:
            training_data=lambda ids:[mixed[sid] for sid in ids] if kind=='mixed' else take(kind,ids)
            result=fit(training_data(train_ids),seed,spec,radial)
            row={'method':method,'training_measurements':kind,'result':result,
                 'test':{k:evaluate(result,take(k,test_ids)) for k in groups},'folds':[]}
            for held,fold_seed in zip(folds,fold_seeds,strict=True):
                training=[sid for sid in train_ids if sid not in held]
                fitted=fit(training_data(training),fold_seed,spec,radial)
                row['folds'].append({'held_ids':held,'result':fitted,
                                     'metrics':{k:evaluate(fitted,take(k,held)) for k in groups}})
            row['cv']={k:{'rms_px':float(np.sqrt(np.mean([v*v for f in row['folds'] for v in f['metrics'][k]['per_pose_rms_px'].values()]))) } for k in groups}
            row['all_fits_converged']=bool(result['converged'] and all(f['result']['converged'] for f in row['folds']))
            row['any_fit_at_bounds']=bool(result['parameters_at_bounds'] or any(f['result']['parameters_at_bounds'] for f in row['folds']))
            rows.append(row);write_json(out/'comparison.json',rows)
            write_json(out/(method+'.json'),result)
            print(method,'test outer/internal',*[round(row['test'][k]['rms_px'],4) for k in groups],
                  'CV outer/internal',*[round(row['cv'][k]['rms_px'],4) for k in groups],flush=True)
        summary={'rows':[{'method':r['method'],'test':{k:r['test'][k]['rms_px'] for k in groups},
                  'cv':{k:r['cv'][k]['rms_px'] for k in groups},'all_fits_converged':r['all_fits_converged'],
                  'any_fit_at_bounds':r['any_fit_at_bounds']} for r in rows],
                  'internal_points_per_pose':{o['sample_id']:len(o['uv']) for o in internal},
                  'status':'diagnostic_comparison_only','physical_accuracy':'unverified'}
        write_json(out/'summary.json',summary)
        write_json(source/'diagnostics/direct_multipoint_review.json',{'output':str(out)})
    except BaseException as exc:
        write_json(out/'failure.json',{'error':str(exc)});raise
    print('OUTPUT='+str(out));return out


def main():
    p=parser(__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--processed',required=True,type=Path)
    a=p.parse_args()
    if a.config:p.error('Uses recorded configuration; no hardware configuration override')
    compare(a.run,a.processed)


if __name__=='__main__':run(main)
