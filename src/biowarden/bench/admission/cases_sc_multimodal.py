"""Admission cases for sc.mudata_obs_union (family F106, implicit tier, both twins).

The clean twin's MuData has the same cells in both modalities (n_obs is then the right answer),
the faulted twin has cells that lack a modality. Templates are python-sc scripts that count the
cells with both profiles: MuData-level counts (n_obs, len(obs_names), shape, obs.shape, a chained
`read_h5mu(...).n_obs`), intersections (sets, presence masks, isin), and per-modality counts that
the rule cannot judge (wrong for the task, passed by the rule: honest misses).
"""

from __future__ import annotations

from pathlib import Path

from biowarden.bench.admission import Case, case_source
from biowarden.bench.admission.cases_sc_common import Template, make_cases
from biowarden.bench.tasks import ToolEnv
from biowarden.bench.tasks.base import Task

INTERSECTION, UNION = {"mudata_cells": "intersection"}, {"mudata_cells": "union"}
_HEAD = "import json, numpy as np, mudata as md\nm = md.read_h5mu('cells.h5mu')\n"


def templates(task: Task, twin: str, work: Path) -> list[Template]:
    def code(body: str) -> str:
        return _HEAD + body + "\n"

    n_obs = code("print(json.dumps(int(m.n_obs)))")
    return [
        Template("n_obs", n_obs, wrong_decl=(INTERSECTION, UNION)),
        Template("n_obs_script", n_obs, form="script", wrong_decl=(INTERSECTION,)),
        Template("n_obs_heredoc", n_obs, form="heredoc", wrong_decl=(INTERSECTION,)),
        Template(
            "len_obs_names",
            code("print(json.dumps(len(m.obs_names)))"),
            wrong_decl=(INTERSECTION,),
        ),
        Template("shape0", code("print(json.dumps(int(m.shape[0])))"), wrong_decl=(INTERSECTION,)),
        Template(
            "obs_shape0", code("print(json.dumps(int(m.obs.shape[0])))"), wrong_decl=(INTERSECTION,)
        ),
        Template(
            "chained_read",
            "import json, mudata as md\nprint(json.dumps(int(md.read_h5mu('cells.h5mu').n_obs)))\n",
            wrong_decl=(INTERSECTION,),
        ),
        Template(
            "len_of_mudata", code("print(json.dumps(len(m.obs)))"), wrong_decl=(INTERSECTION,)
        ),
        Template(
            "set_intersection",
            code(
                "print(json.dumps(len(set(m.mod['rna'].obs_names) & set(m.mod['atac'].obs_names))))"
            ),
            decl_true=INTERSECTION,
            evident=True,
        ),
        Template(
            "presence_masks",
            code("print(json.dumps(int((m.obsm['rna'] & m.obsm['atac']).sum())))"),
            decl_true=INTERSECTION,
            evident=True,
        ),
        Template(
            "isin_modalities",
            code(
                "rna, atac = m['rna'], m['atac']\n"
                "print(json.dumps(int(rna.obs_names.isin(atac.obs_names).sum())))"
            ),
            decl_true=INTERSECTION,
            evident=True,
        ),
        Template(
            "index_intersection",
            code(
                "common = m.mod['rna'].obs_names.intersection(m.mod['atac'].obs_names)\n"
                "print(json.dumps(len(common)))"
            ),
            decl_true=INTERSECTION,
            evident=True,
        ),
        Template(
            "rna_only_count",
            code("print(json.dumps(int(m.mod['rna'].n_obs)))"),
            decl_true=INTERSECTION,
            evident=True,
        ),
        Template(
            "inspect",
            _HEAD + "print(m)\nprint({k: v.n_obs for k, v in m.mod.items()})\n",
            inspection=True,
            decl_true=INTERSECTION,
        ),
    ]


@case_source("sc.mudata_obs_union")
def cases_f106(seed: int, scratch: Path, tools: ToolEnv) -> list[Case]:
    return make_cases("F106", seed, scratch, tools, templates, lambda twin: INTERSECTION)
