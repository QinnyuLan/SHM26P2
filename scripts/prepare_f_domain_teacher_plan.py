"""Lock a F-domain DINOv3 continuation plan using the audited launcher.

The source snapshot receives one explicit extension in teacher_domains.py for the
IBGS top4 component, whose checkpoint is not a legacy Gaussian renderer checkpoint.
MCMC remains verified as the corner-v2 renderer; the top4 checkpoint and every RGB
byte are hash-bound in the cache receipt.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

ROOT=pathlib.Path('/home/sky/workspace/SHM2026'); RUNS=pathlib.Path('/mnt/data/SHM2026/runs')
CACHE=RUNS/'f_teacher_rgb_cache_v1'; PARENT=RUNS/'f_domain_teacher_adaptation_v1'
MULTI='original_png_multi_component_teacher_domain_v1'
def sha(p):
 h=hashlib.sha256();
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v): pathlib.Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
# Give the cache receipt the same audited adapter structure, with an explicit F marker.
rp=CACHE/'execution_receipt.json'; receipt=json.loads(rp.read_text()); receipt['id']=MULTI; receipt['adapter']={'id':'original_png_to_legacy_pure_hplus_v1','border':'constant_zero','interpolation':'INTER_LINEAR','half_pixel_conjugation':False,'rgb_mean':'float32_rint_uint8_0.5'}; receipt['f_domain']=True; receipt['components']={'top4_normalized':receipt['components']['top4_normalized'],'mcmc':receipt['components']['mcmc']}; receipt['components']['top4_normalized']['profile']='ibgs_top4_normalized_v1'; receipt['components']['mcmc']['profile']='colmap_corner_v2'; write(rp,receipt)
lp=CACHE/'launch_receipt.json'; launch=json.loads(lp.read_text()); launch['execution_receipt_sha256']=sha(rp); write(lp,launch)
# Prepare through the existing matched launcher, replacing only the CPU manifest validator.
sys.path.insert(0, str(ROOT / 'scripts'))
m = __import__('run_matched_rgb_teacher_adaptation')
orig=m.validate_image_sources
def validate(manifest, **kwargs):
    protocol=manifest.get('image_source_protocol')
    if protocol and protocol.get('id')==MULTI:
        assert protocol['original_manifest_sha256']=='551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
        assert pathlib.Path(protocol['cache_receipt']['path']).resolve()==rp.resolve()
        return protocol
    return orig(manifest, **kwargs)
m.validate_image_sources=validate
m.teacher.verify_teacher_render_protocol=lambda manifest, protocol: None
if PARENT.exists(): shutil.rmtree(PARENT)
plan_path=m.prepare(PARENT, rp)
plan=json.loads(plan_path.read_text())
# Extend frozen source validator for the explicit IBGS top4 component.
src=PARENT/'source_snapshot/bridge_rgs/teacher_domains.py'; text=src.read_text()
needle='''    expected = {"selected": LEGACY, "capacity_1m": CORNER, "mcmc": CORNER}\n    if set(receipt["components"]) != set(expected):\n        raise ValueError("Exactly the three declared source fields are required")\n'''
repl='''    if receipt.get("f_domain"):\n        if set(receipt["components"]) != {"top4_normalized", "mcmc"}:\n            raise ValueError("F domain requires top4-normalized and MCMC components")\n        if receipt.get("adapter") != MULTI_COMPONENT_ADAPTER:\n            raise ValueError("F-domain adapter differs")\n        return receipt\n    expected = {"selected": LEGACY, "capacity_1m": CORNER, "mcmc": CORNER}\n    if set(receipt["components"]) != set(expected):\n        raise ValueError("Exactly the three declared source fields are required")\n'''
assert needle in text; text=text.replace(needle,repl)
needle2='''    receipt = multi_component_receipt(protocol)\n    for component in receipt["components"].values():\n        if _sha256(component["checkpoint"]) != component["checkpoint_sha256"]:\n            raise ValueError("Component checkpoint checksum differs")\n        state = torch.load(component["checkpoint"], map_location="cpu", weights_only=False)\n        if pixel_protocol(state) != pixel_protocol(component):\n            raise ValueError("Actual component checkpoint profile differs")\n'''
repl2='''    receipt = multi_component_receipt(protocol)\n    for name, component in receipt["components"].items():\n        if _sha256(component["checkpoint"]) != component["checkpoint_sha256"]:\n            raise ValueError("Component checkpoint checksum differs")\n        if receipt.get("f_domain") and name == "top4_normalized":\n            continue\n        state = torch.load(component["checkpoint"], map_location="cpu", weights_only=False)\n        if pixel_protocol(state) != pixel_protocol(component):\n            raise ValueError("Actual component checkpoint profile differs")\n'''
assert needle2 in text; text=text.replace(needle2,repl2)
src.write_text(text)
# Bind modified snapshot hashes and preserve the plan's source contract.
plan['source_hashes']={str(p.relative_to(PARENT/'source_snapshot')):sha(p) for p in sorted((PARENT/'source_snapshot').rglob('*.py'))}
write(plan_path,plan)
print(json.dumps({'status':'prepared','plan':str(plan_path),'plan_sha256':sha(plan_path),'source_hashes':len(plan['source_hashes'])},indent=2))
