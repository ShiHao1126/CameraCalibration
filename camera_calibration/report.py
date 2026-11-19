"""Persist solve, frozen holdout evaluation and a portable HTML report."""
from datetime import datetime
from pathlib import Path
import html
import json
from .dataset import read_json,write_json
from .solver import solve
from .validation import evaluate
from .detection import filter_capture
from .models import jsonable


def render_report(folder,result,validation):
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Camera calibration</title>
<style>body{font:16px system-ui;max-width:1050px;margin:36px auto;padding:0 24px;line-height:1.6}pre{overflow:auto;background:#f3f4f5;padding:16px}h1{font-size:26px}</style>
<h1>相机标定结果</h1><p>相机：%s · 求解：%s · 留出验证：%s · 物理精度：%s</p>
<p>T_W_C 将原生 RGB 相机坐标转换到 Flexiv world 坐标。仅在配置了 T_base_world 后导出 T_B_C。</p>
<p>像素误差和参数可辨识性不能证明真实定位达到 1 mm。</p><h2>结果</h2><pre>%s</pre><h2>验证</h2><pre>%s</pre></html>'''
    content=doc % tuple(html.escape(str(v)) for v in [result['camera']['serial'],result['status_solve'],
        validation['status_internal_validation'],validation['status_physical_accuracy'],
        json.dumps(jsonable(result),ensure_ascii=False,indent=2),json.dumps(jsonable(validation),ensure_ascii=False,indent=2)])
    (Path(folder)/'report.html').write_text(content)


def calibrate_run(run,method='hybrid',multi=True):
    if method=='hybrid':
        from .hybrid import calibrate_hybrid
        return calibrate_hybrid(run,multi=multi)
    run=Path(run); manifest=read_json(run/'manifest.json'); cfg=manifest['config']
    if manifest['status']!='captured': raise ValueError('Calibration requires a completed capture run')
    if manifest.get('filter_status') in ('pending','running','failed') or not (run/'observations.json').is_file():
        filter_capture(run)
    observations=read_json(run/'observations.json')
    train=[o for o in observations if o['split']=='train']; test=[o for o in observations if o['split']=='test']
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    out=run/'calibration'/f'{stamp}_{method}'; out.mkdir(parents=True)
    try:
        if len(train)<cfg['solver']['min_train_poses'] or len(test)<3:
            raise ValueError(f'Insufficient valid poses after filtering: train={len(train)} '
                             f'(need {cfg["solver"]["min_train_poses"]}), test={len(test)} (need 3). '
                             'Inspect filter_report.json and collect more diverse poses; no calibration result was produced.')
        result=solve(train,manifest['camera'],cfg,method=method,multi=multi)
        validation=evaluate(result,test,cfg)
        result['status_internal_validation']=validation['status_internal_validation']
        result['status_physical_accuracy']=validation['status_physical_accuracy']
        result['robot_serial']=manifest['robot']['serial']; result['config']=cfg
        result['capture_run']=str(run.resolve())
        write_json(out/'result.json',result); write_json(out/'validation.json',validation)
        render_report(out,result,validation)
    except BaseException as e:
        write_json(out/'failure.json',{'status_solve':'failed','error':str(e)})
        raise
    return out,result,validation
