"""Build an offline, per-pose/per-frame corner inspection page from saved RGB data."""
from _common import parser,run
from pathlib import Path
import json
import os
import numpy as np
from camera_calibration.dataset import read_json,read_jsonl,write_json
from camera_calibration.geometry import CameraModel,marker_points,inverse
from camera_calibration.multipoint import grid_points


def build(source,processed):
    source=Path(source).resolve(); processed=Path(processed).resolve()
    manifest=read_json(source/'manifest.json'); out=source/'diagnostics/corner_review'; out.mkdir(parents=True,exist_ok=True)
    result=read_json(processed/'result.json')
    primary='hybrid' if result.get('method')=='hybrid' else 'joint_current_burst'
    models=[{**result,'method':primary,'corner_source':'direct_internal' if primary=='hybrid' else 'r3kit'}]
    comparison=source/'diagnostics/r3kit_full_comparison/comparison.json'
    comparisons=read_json(comparison) if comparison.is_file() else []
    models=[r for r in comparisons if r['method']!='joint_current_burst']+models
    investigation=source/'diagnostics/residual_investigation'
    feature_source=processed if primary=='hybrid' else investigation
    if primary=='hybrid':
        current=read_json(processed/'comparison.json')
        for r in current.get('r3kit',[]):
            models.append({**r,'method':'production_'+r['method']})
    experiments=read_json(investigation/'camera_complexity_selection.json') if (investigation/'camera_complexity_selection.json').is_file() else []
    candidates=[]
    if experiments:
        selected=min(experiments,key=lambda x:x['blocked_training_cv_rms_px'])
        optimistic=min(experiments,key=lambda x:x['test']['rms_px'])
        for label,row in [('diagnostic_selected_by_training_cv',selected),('diagnostic_lowest_seen_test_not_selected',optimistic)]:
            models.append({**row['fit'],'method':label,'corner_source':row['corners']})
            candidates.append({'method':label,'corners':row['corners'],'model':row['model'],
                               'cv_rms_px':row['blocked_training_cv_rms_px'],'test_rms_px':row['test']['rms_px']})
    multipoint=[]
    pointer=source/'diagnostics/direct_multipoint_review.json'
    if pointer.is_file():
        multiout=Path(read_json(pointer)['output'])
        if not multiout.resolve().is_relative_to(source):raise ValueError('Multipoint output escapes source capture')
        multipoint=read_json(multiout/'comparison.json')
        for row in multipoint:
            models.append({**row['result'],'method':row['method'],
                           'corner_source':'edge_lines' if row['training_measurements']=='outer' else 'direct_internal'})
    measurements={o['sample_id']:o for o in read_json(processed/'observations.json')}
    samples=[]
    for target in manifest['plan']['samples']:
        folder=source/'samples'/target['id']; frames=read_jsonl(folder/'frames.jsonl')
        revised=read_json(processed/'detections'/f'{target["id"]}.json')
        original=read_json(folder/'detections.json') if (folder/'detections.json').is_file() else [{'detection':None} for _ in frames]
        extra={}
        for key in ['edge','internal']:
            f=feature_source/(key+'_detections')/f'{target["id"]}.json'
            extra[key]=read_json(f) if f.is_file() else [None]*len(frames)
        row={'id':target['id'],'split':target['split'],'valid':target['id'] in measurements,'frames':[]}
        for frame_index,(raw,new,old) in enumerate(zip(frames,revised,original,strict=True)):
            predictions={}; errors={};internal_predictions={}
            frame_internal=extra['internal'][frame_index]
            internal_fit=frame_internal.get('internal_fit') if frame_internal else None
            local_points=grid_points(internal_fit['grid_points']) if internal_fit and internal_fit['accepted'] else None
            for model in models:
                C=inverse(np.asarray(model['T_W_C']))@np.asarray(raw['actual']['T_W_E'])@np.asarray(model['T_E_M'])
                points=marker_points()@C[:3,:3].T+C[:3,3]
                try: uv=CameraModel(model['camera']).project(points)
                except ValueError: continue
                predictions[model['method']]=uv.tolist()
                if local_points is not None:
                    xyz=local_points@C[:3,:3].T+C[:3,3]
                    internal_predictions[model['method']]=CameraModel(model['camera']).project(xyz).tolist()
                if new['detection']:
                    errors[model['method']]=float(np.sqrt(np.mean(np.sum((uv-np.asarray(new['detection']['uv']))**2,axis=1))))
            row['frames'].append({'image':os.path.relpath(folder/raw['image'],out),
                'name':raw['image'],'r3kit':new['detection'],'old':old['detection'],
                'edge':extra['edge'][frame_index],'internal':extra['internal'][frame_index],
                'reason':new['rejection'],'predictions':predictions,'internal_predictions':internal_predictions,'rms_px':errors})
        samples.append(row)
    data={'camera':manifest['camera'],'source':str(source),'samples':samples,'default_model':primary,
          'models':[m['method'] for m in models],
          'model_corner_source':{m['method']:m.get('corner_source','r3kit') for m in models},'candidates':candidates,
          'multipoint':[{'method':r['method'],'test':{k:r['test'][k]['rms_px'] for k in ['outer','internal']},
                        'cv':{k:r['cv'][k]['rms_px'] for k in ['outer','internal']},
                        'converged':r['all_fits_converged'],'at_bounds':r['any_fit_at_bounds']} for r in multipoint],
          'comparison':[{'method':c['method'],'train_count':c['train_count'],'test_count':c['test_count'],
             'train_rms_px':c['train']['rms_px'],'test_rms_px':c['test']['rms_px'],
             'upstream_per_image_rms_px':c.get('upstream_original_result',{}).get('error',[None])[0]} for c in comparisons]}
    write_json(out/'review_data.json',data)
    page=TEMPLATE.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (out/'index.html').write_text(page)
    print(out/'index.html'); return out/'index.html'


TEMPLATE=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>标定角点逐帧检查</title>
<style>
body{margin:24px;font:15px system-ui;background:#f4f5f6;color:#202326}h1{font-size:24px;margin-bottom:8px}h2{font-size:18px}
button,select{font:inherit;padding:7px;margin:3px;border:1px solid #abb2b9;background:white;border-radius:4px}label{margin-right:16px}
.bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:12px 0}svg{width:100%;height:65vh;background:#111;display:block}
.note{color:#525c65;line-height:1.6}.legend span{margin-right:20px}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}#status{font-family:monospace;white-space:pre-wrap;margin:10px 0;line-height:1.6}
#grid{display:flex;flex-wrap:wrap}#grid button[data-active=true]{background:#243e54;color:white}#grid button[data-valid=false]{border-style:dashed;color:#8b3030}
</style>
<h1>标定角点逐帧检查 · D415 104122061018</h1>
<p class="note">绿点是当前选择的外角点，混合方法采用边缘细化角点；橙色叉号是固定相机/安装外参结合该帧实际法兰位姿预测的位置。编号 0–3 为解码后的角点顺序。原图未修改。未检出/不满足有效帧数量的点也完整显示。</p>
<div class="bar"><button id="prev">← 上一点</button><select id="pose"></select><button id="next">下一点 →</button>
<span>帧</span><select id="frame"></select><span>预测模型</span><select id="model"></select></div>
<div class="bar"><label><input id="zoom" type="checkbox" checked>放大标记区域</label>
<label>角点来源 <select id="measurement"><option value="r3kit">r3kit 原角点</option><option value="edge">试验：边缘直线交点</option><option value="internal">旧试验：内部图案拟合出的外角点</option><option value="direct_internal">内部实测点直接优化</option></select></label>
<label><input id="old" type="checkbox">显示旧 SUBPIX 结果</label><label><input id="raw" type="checkbox">显示细化前检测</label>
<label><input id="prediction" type="checkbox" checked>显示外参预测</label></div>
<div class="bar"><label><input id="internal_points" type="checkbox" checked>显示内部实测点与预测</label><label><input id="internal_labels" type="checkbox">标注内部点网格编号</label></div>
<div class="legend"><span style="color:#00874e">● 当前选择的图像角点</span><span style="color:#b75e00">× 外参预测</span>
<span style="color:#b72860">虚线：旧 SUBPIX</span><span style="color:#167b9c">青点：内部实测；小橙叉：内部预测</span></div>
<div id="status"></div><svg id="view" xmlns="http://www.w3.org/2000/svg"></svg>
<p class="note">快捷键：←/→ 切换采样点，↑/↓ 切换 burst 帧。观察橙色预测前，可先关闭它，仅确认绿点是否落在 80 mm 标记黑色外边界四角。缺少检测时显示完整画面。</p>
<div id="grid"></div><h2>相同指标下的方法对照</h2><div id="table"></div>
<h2>内部多点直接优化 · 相同分组、相同评价点</h2><div id="multipoint"></div><p class="note">内部实测点来自每帧图案交界处的亚像素定位，直接参与共享外参优化；不再用每张图的平面映射生成四个虚拟外角点。所有方法都在同一组内部实测点、同一组边缘外角点上分别评分。比较时必须比较同一列，不能把内部点 RMS 与另一方法的外角点 RMS 混比。34 个训练姿态、6 个原测试姿态及训练集连续五组划分不变；本页逐帧值与表内 burst 聚合值不同。</p>
<h2>剩余误差试验</h2><div id="experiments"></div><p class="note">候选采用相同 80 mm 标记、相同样本和划分。分组交叉验证只在原训练集内进行；测试误差最低的模型并不一定可靠。此处保留历史诊断候选；当前默认混合方法由上方模型选项标明。选择预测模型时会自动切换对应的角点来源，绿色仍表示图像测量而非机器人预测。</p>
<p class="note">固定外参误差：不重新拟合测试图像的板姿态，直接用实际机器人位姿预测角点。r3kit 自带图像 RMS 则允许每张训练图独立拟合板姿态，不能与前者直接等同。first_frame 每个点使用第 0 帧；burst 使用已有有效帧聚合，样本集合略有不同。测试成员保持原划分，但这些图像已被用于排障，不能当作全新独立验收。像素误差也不等于毫米精度。</p>
<details><summary>数据来源</summary><pre id="source"></pre></details>
<script>
const D=__DATA__;const $=id=>document.getElementById(id);const initial=new URLSearchParams(location.hash.slice(1));let pose=Math.max(0,D.samples.findIndex(s=>s.id===initial.get('pose'))),frame=Number(initial.get('frame')||0);
$('old').checked=initial.get('old')==='1';
const S='http://www.w3.org/2000/svg';function node(name,attrs,text){const e=document.createElementNS(S,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
const names={hybrid:'当前默认：内部点＋外角点混合优化',production_r3kit_same_burst:'当前对照：r3kit 原求解器·相同连拍',production_r3kit_full_first_frame:'当前对照：r3kit 完整原流程·首帧',edge_radial_baseline:'对照：边缘外角点＋径向畸变（原 1.28px）',direct_internal_fixed:'内部实测点直接优化 · 固定相机模型',direct_internal_radial:'内部实测点直接优化＋径向畸变',direct_internal_and_edge_radial:'内部实测点＋外角点联合优化（各半权重）',joint_current_burst:'原主方法：四角点联合优化（1.47px）',diagnostic_selected_by_training_cv:'诊断候选：边缘定位＋径向畸变（1.28px）',diagnostic_lowest_seen_test_not_selected:'未选用：测试最低但分组验证变差'};
function measurementSource(key){return key==='edge_lines'?'edge':key==='internal_grid'?'internal':key==='direct_internal'?'direct_internal':'r3kit'}
D.samples.forEach((s,i)=>{$('pose').append(option(i,`${s.id} · ${s.split} · ${s.valid?'有效':'未通过图像筛选'}`));const b=document.createElement('button');b.textContent=s.id;b.dataset.valid=s.valid;b.onclick=()=>{pose=i;frame=0;render()};$('grid').append(b)});
D.models.forEach(m=>$('model').append(option(m,names[m]||m)));$('model').value=initial.get('model')||D.default_model;$('source').textContent=D.source;
const initialSource=D.model_corner_source[$('model').value];$('measurement').value=initial.get('measurement')||measurementSource(initialSource);
function polygon(points,color,dashed,labels){if(!points)return;const a={points:points.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':1,'vector-effect':'non-scaling-stroke'};if(dashed)a['stroke-dasharray']='5 3';$('view').append(node('polygon',a));if(labels)points.forEach((p,i)=>{$('view').append(node('circle',{cx:p[0],cy:p[1],r:1.3,fill:color}));$('view').append(node('text',{x:p[0]+2,y:p[1]-2,fill:color,'font-size':6,'paint-order':'stroke',stroke:'#111','stroke-width':.3},i));});}
function render(){const s=D.samples[pose];frame=Math.max(0,Math.min(frame,s.frames.length-1));const f=s.frames[frame];$('pose').value=pose;$('frame').replaceChildren(...s.frames.map((_,i)=>option(i,i)));$('frame').value=frame;
 const v=$('view');v.replaceChildren();v.append(node('image',{href:f.image,x:0,y:0,width:D.camera.width,height:D.camera.height}));
 const measurement=$('measurement').value;const observed=measurement==='r3kit'?f.r3kit:measurement==='direct_internal'?f.edge?.detection:f[measurement]?.detection;const p=observed?.uv,old=f.old?.uv;let box=[0,0,D.camera.width,D.camera.height];if($('zoom').checked&&(p||old)){let points=p||old;let xs=points.map(v=>v[0]),ys=points.map(v=>v[1]);let x=Math.max(0,Math.min(...xs)-45),y=Math.max(0,Math.min(...ys)-45);box=[x,y,Math.min(D.camera.width-x,Math.max(...xs)-x+45),Math.min(D.camera.height-y,Math.max(...ys)-y+45)];}v.setAttribute('viewBox',box.join(' '));
 if($('old').checked)polygon(old,'#ff528e',true,false);if($('raw').checked)polygon(f.r3kit?.detector_uv,'#00d7fa',false,false);polygon(p,'#28ff8b',false,true);
 const pred=f.predictions[$('model').value];if($('prediction').checked&&pred){pred.forEach(a=>{v.append(node('path',{d:`M ${a[0]-2} ${a[1]-2} l 4 4 M ${a[0]-2} ${a[1]+2} l 4 -4`,stroke:'#ff9c28','stroke-width':1,'vector-effect':'non-scaling-stroke'}));});}
 const observedInternal=f.internal?.internal_fit?.accepted?f.internal.internal_fit.image_points:null;const grids=f.internal?.internal_fit?.grid_points;const predictedInternal=f.internal_predictions?.[$('model').value];
 if($('internal_points').checked&&observedInternal){observedInternal.forEach((a,i)=>{const dot=node('circle',{cx:a[0],cy:a[1],r:.85,fill:'#00d7fa',stroke:'#111','stroke-width':.15});dot.append(node('title',{},`网格 (${grids[i].join(',')}) · 实测 (${a[0].toFixed(2)}, ${a[1].toFixed(2)})`));v.append(dot);if($('internal_labels').checked)v.append(node('text',{x:a[0]+1.5,y:a[1]-1.5,fill:'#00d7fa','font-size':3.5,'paint-order':'stroke',stroke:'#111','stroke-width':.2},grids[i].join(',')));});
 if($('prediction').checked&&predictedInternal)predictedInternal.forEach(a=>v.append(node('path',{d:`M ${a[0]-1.3} ${a[1]-1.3} l 2.6 2.6 M ${a[0]-1.3} ${a[1]+1.3} l 2.6 -2.6`,stroke:'#ff9c28','stroke-width':1,'vector-effect':'non-scaling-stroke'})));}
 const err=p&&pred?Math.sqrt(p.reduce((s,a,i)=>s+(a[0]-pred[i][0])**2+(a[1]-pred[i][1])**2,0)/4):undefined;let jump=p&&old?Math.max(...p.map((a,i)=>Math.hypot(a[0]-old[i][0],a[1]-old[i][1]))):null;
 const internalError=observedInternal&&predictedInternal?Math.sqrt(observedInternal.reduce((sum,a,i)=>sum+(a[0]-predictedInternal[i][0])**2+(a[1]-predictedInternal[i][1])**2,0)/observedInternal.length):null;
 $('status').textContent=`点 ${s.id} / ${s.split} · 帧 ${frame} · ${p?'识别 ID 0':'当前来源无测量：'+(f.reason||'该点未进入原有效样本集')}\n当前帧外角点 RMS：${err===undefined?'—':err.toFixed(3)+' px'} · 内部 ${observedInternal?.length??0} 点 RMS：${internalError===null?'—':internalError.toFixed(3)+' px'} · 新旧外角点最大差：${jump===null?'—':jump.toFixed(3)+' px'}`;
 [...$('grid').children].forEach((b,i)=>b.dataset.active=i===pose);history.replaceState(null,'',`#pose=${s.id}&frame=${frame}&old=${$('old').checked?'1':'0'}&model=${$('model').value}&measurement=${measurement}`);}
$('prev').onclick=()=>{pose=(pose-1+D.samples.length)%D.samples.length;frame=0;render()};$('next').onclick=()=>{pose=(pose+1)%D.samples.length;frame=0;render()};$('pose').onchange=e=>{pose=+e.target.value;frame=0;render()};$('frame').onchange=e=>{frame=+e.target.value;render()};['measurement','zoom','old','raw','prediction','internal_points','internal_labels'].forEach(id=>$(id).onchange=render);
$('model').onchange=()=>{$('measurement').value=measurementSource(D.model_corner_source[$('model').value]);render()};
document.onkeydown=e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if(e.key==='ArrowUp'){frame--;render()}if(e.key==='ArrowDown'){frame++;render()}if(e.key.startsWith('Arrow'))e.preventDefault();};
const table=document.createElement('table');let tr=table.insertRow();['方法','训练点/测试点','训练固定外参 RMS (px)','测试固定外参 RMS (px)','r3kit 图像拟合 RMS (px)'].forEach(t=>{const th=document.createElement('th');th.textContent=t;tr.append(th)});D.comparison.forEach(r=>{const t=table.insertRow();[r.method,`${r.train_count}/${r.test_count}`,r.train_rms_px.toFixed(3),r.test_rms_px.toFixed(3),r.upstream_per_image_rms_px?.toFixed(3)??'—'].forEach(x=>t.insertCell().textContent=x)});$('table').append(table);
const experiments=document.createElement('table');let h=experiments.insertRow();['选择依据','角点/相机模型','训练分组 CV RMS (px)','原测试集 RMS (px)'].forEach(t=>{const th=document.createElement('th');th.textContent=t;h.append(th)});D.candidates.forEach(r=>{const t=experiments.insertRow();[r.method,r.corners+' / '+r.model,r.cv_rms_px.toFixed(3),r.test_rms_px.toFixed(3)].forEach(x=>t.insertCell().textContent=x)});$('experiments').append(experiments);render();
const multi=document.createElement('table');const mh=multi.insertRow();['方法','原测试：外角点 RMS (px)','原测试：内部点 RMS (px)','分组 CV：外角点 RMS (px)','分组 CV：内部点 RMS (px)'].forEach(t=>{const th=document.createElement('th');th.textContent=t;mh.append(th)});D.multipoint.forEach(r=>{const t=multi.insertRow();[names[r.method]||r.method,r.test.outer.toFixed(3),r.test.internal.toFixed(3),r.cv.outer.toFixed(3),r.cv.internal.toFixed(3)].forEach(x=>t.insertCell().textContent=x)});$('multipoint').append(multi);
</script></html>'''


def main():
    p=parser(__doc__); p.add_argument('--run',type=Path,required=True); p.add_argument('--processed',type=Path,required=True)
    a=p.parse_args(); build(a.run,a.processed)


if __name__=='__main__': run(main)
