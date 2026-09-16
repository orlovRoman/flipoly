import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
m_path = os.path.join(_ROOT, 'out', 'RUN_MANIFEST.json')
with open(m_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

# Binding SHA = frozen feature audit spec (lgbm_feature_audit_v1.yaml).
# Direction spec c36f0de5... is deliberately NOT the binding SHA for the
# feature audit; do not revert this line.
manifest['spec_sha'] = 'ca4dd2fdb7ea6b65e084cb4c59941bcb65741519b385b1fc36e4c5a971147295'
manifest['spec_commit'] = 'b37371af0108139da5143cf6a51fd57bae1758a3'
manifest['spec_source'] = 'research/lgbm-feature-audit-v1 spec/lgbm_feature_audit_v1.yaml (frozen v1.0.0)'

with open(m_path, 'w', encoding='utf-8', newline='') as f:
    json.dump(manifest, f, indent=1, default=str)
    f.write('\n')

print('RUN_MANIFEST updated:')
with open(m_path, 'r', encoding='utf-8') as f:
    print(f.read())