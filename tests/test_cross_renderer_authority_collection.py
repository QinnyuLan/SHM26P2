"""CPU collection contracts; no dataset image or label payloads are read."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent/'collect_cross_renderer_authority.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1]/'scripts/collect_cross_renderer_authority.py'
spec = importlib.util.spec_from_file_location('authority_collection_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def population():
    return {'views': [{'name':f'{i:03d}.png','split':'train','mask_path':f'/not/open/{i}.png' if i<259 else None}
                      for i in range(350)]}


def test_fixed_uniform_labeled_indices_independent_of_input_order():
    manifest = population()
    manifest['views'].reverse()
    views, indices = m.select_views(manifest)
    assert indices == [0,17,34,51,68,86,103,120,137,154,172,189,206,223,240,258]
    assert [v['name'] for v in views] == [f'{i:03d}.png' for i in indices]


@pytest.mark.parametrize('failure', ['duplicate','missing_label','wrong_split'])
def test_population_change_rejected(failure):
    manifest = population()
    if failure == 'duplicate':
        manifest['views'][-1]['name'] = manifest['views'][0]['name']
    elif failure == 'missing_label':
        manifest['views'][0]['mask_path'] = None
    else:
        manifest['views'][-1]['split'] = 'val'
    with pytest.raises(ValueError,match='350 TRAIN'):
        m.select_views(manifest)


def test_target_sha_is_inherited_without_open_hash_or_existence_check(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Collector must not hash/read target')
    monkeypatch.setattr(m,'sha',forbidden)
    view={'mask_path':'/missing_target.png','valid_path':'/missing_valid.png'}
    old={view['mask_path']:'a'*64,view['valid_path']:'b'*64}
    records=m.target_declarations(view,old)
    assert records['mask']['sha256']=='a'*64 and records['valid']['path']==view['valid_path']
    with pytest.raises(ValueError,match='historical'):
        m.target_declarations(view,{})


def test_64_probability_barrier_and_failed_receipt_cannot_emit_analysis():
    views=[{'name':f'{i:03d}.png','mask':{},'valid':{}} for i in range(16)]
    records=[{'name':v['name'],'split':'train',**{k:{'path':f'{k}/{i}.npy'} for k in m.KINDS}}
             for i,v in enumerate(views)]
    report={'status':'completed','inputs_and_sources_unchanged':True,'predictions':records}
    result=m.analysis_declaration(report,views,{'path':'receipt','sha256':'hash'},{'fixed':True})
    assert result['status']=='predictions_completed' and result['specification']=={'fixed':True}
    assert len(result['views'])==16 and 'mask' in result['views'][0]
    with pytest.raises(ValueError,match='Failed/partial'):
        m.analysis_declaration(dict(report,status='failed'),views,{})
    del records[-1]['Tm']
    with pytest.raises(ValueError,match='64 unique'):
        m.analysis_declaration(report,views,{})


def test_saved_soft_canvas_has_no_argmax_quantization_or_backwarp(tmp_path):
    probabilities=np.array([[[.1,.2,.3,.25,.15],[.9,.025,.025,.025,.025]]],np.float32)
    record=m.save_probability(tmp_path/'p.npy',probabilities)
    saved=np.load(record['path'],allow_pickle=False)
    np.testing.assert_array_equal(saved,probabilities)
    assert saved.dtype==np.float32 and record['shape']==[1,2,5]
    with pytest.raises(ValueError,match='FP32'):
        m.save_probability(tmp_path/'wrong.npy',probabilities.astype(np.float64))
    with pytest.raises(ValueError,match='probabilities'):
        m.save_probability(tmp_path/'zero.npy',np.zeros_like(probabilities))
