"""Table IV variants. A0 is the full model; A1-A3 remove a module; A4/A5 mask sequence-external evidence.

A4 (w/o EA)  zeros external supporting-node attributes in path and background summaries; denominators kept
A5 (w/o EPS) path counts, transition mass and path indicators use only paths inside S_c
"""
PROTOCOL_ID = 'scope_ablation_final_tuned_v4_20261022'
LEGACY_PROTOCOL_ID = 'scope_ablation_walk_tcb_v4_20261021'
COMMON = {'execution': 'packed'}
VARIANTS = {
    'A0': ('Full', 'relation_source', {}),
    'A1': ('No-R', 'relation_source', {'relations': 'off'}),
    'A3': ('No-S', 'sampling', {'sampling': 'uniform_rw'}),
    'A2': ('No-C (Full-query)', 'attention', {'query': 'context', 'execution': 'dense'}),
    'A4': ('No-External-Attributes', 'external_evidence', {'external_evidence': 'attributes_off'}),
    'A5': ('No-External-Path-Statistics', 'external_evidence', {'external_evidence': 'structure_off'}),
}
GROUPS = {'core': ['A0', 'A1', 'A3', 'A2'], 'all': list(VARIANTS), 'external': ['A4', 'A5']}
GROUPS.update(main=['A0'], controls=['A1', 'A3', 'A2'])
GROUPS.update(relation_source=['A0', 'A1'], sampling=['A0', 'A3'], attention=['A0', 'A2'])
ALIASES = {'Full': 'A0', 'No-R': 'A1', 'No-S': 'A3', 'No-C': 'A2', 'Full-query': 'A2'}
ALIASES.update({'No-External-Attributes': 'A4', 'No-External-Path-Statistics': 'A5'})
DEFINING_DEFAULTS = dict(relations='on', relation_source='graph', sampling='structured', query='center', execution='packed',
                        external_evidence='all')
VARIANT_ARGS = (*DEFINING_DEFAULTS, 'variant')


def flags(variant):
    out = []
    for key, value in {**COMMON, **VARIANTS[variant][2]}.items():
        out.extend(['--' + key.replace('_', '-'), str(value)])
    return out


def expand(names):
    result = []
    for name in names:
        name = ALIASES.get(name, name)
        if name not in GROUPS and name not in VARIANTS:
            raise KeyError(name)
        result.extend(v for v in GROUPS.get(name, [name]) if v not in result)
    return result
