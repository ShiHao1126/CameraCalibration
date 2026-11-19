import numpy as np
import pytest
from conftest import synthetic_observations
from camera_calibration.validation import evaluate
from camera_calibration.dataset import Journal,read_jsonl,write_json,read_json


def test_train_test_leakage_rejected(cfg):
    obs,spec,X,Y=synthetic_observations()
    r={'T_W_C':X.tolist(),'T_E_M':Y.tolist(),'camera':spec,'train_ids':['0'],'status_solve':'ok'}
    with pytest.raises(ValueError,match='overlap'): evaluate(r,[obs[0]],cfg)


def test_absent_reference_never_reports_physical_pass(cfg):
    obs,spec,X,Y=synthetic_observations()
    r={'T_W_C':X.tolist(),'T_E_M':Y.tolist(),'camera':spec,'train_ids':[],'status_solve':'ok'}
    report=evaluate(r,obs,cfg)
    assert report['rms_px']<1e-8 and report['status_physical_accuracy']=='unverified'


def test_atomic_json_and_corrupt_journal(tmp_path):
    p=tmp_path/'meta.json'; write_json(p,{'T':np.eye(4)})
    assert read_json(p)['T'][3][3]==1
    j=tmp_path/'states.jsonl'
    with Journal(j) as log: log.append({'index':0})
    assert read_jsonl(j)==[{'index':0}]
    with j.open('a') as f: f.write('{')
    with pytest.raises(ValueError,match='Incomplete'): read_jsonl(j)
