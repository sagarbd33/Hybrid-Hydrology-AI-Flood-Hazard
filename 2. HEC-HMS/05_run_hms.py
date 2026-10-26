# -*- coding: utf-8 -*-
"""
Step 5 - Run HEC-HMS 4.10 for every scenario (Windows).

HEC-HMS is free software of the US Army Corps of Engineers, Hydrologic
Engineering Center (https://www.hec.usace.army.mil/software/hec-hms/).
The portable Windows build of version 4.10 is expected in HEC-HMS-4.10/
inside this folder (set HMS_DIR in common.py otherwise).

For each project in HMS_Projects/ a small Jython script is written and run
with the HEC-HMS engine (hms.Hms -script <file>), which computes run "Cal"
(the run that uses Calibration.control). HEC-HMS writes the results to
HMS_Projects/<Scenario>/Cal.dss and the log to Cal.log.

Java is called directly with the same options as HEC-HMS.cmd, but with every
path quoted, so the folders may contain spaces (e.g. "Paper 6\\2. HEC-HMS").

Run:  python 05_run_hms.py                  (all scenarios)
      python 05_run_hms.py Ssp245 Ssp585    (selected scenarios)
"""

import os
import subprocess
import sys

from common import HMS_DIR, RUN_NAME, SCENARIOS, project_dir

JYTHON = '''from hms.model import Project
from hms import Hms

project = Project.open(r"{hms_file}")
project.computeRun("{run}")
project.close()
Hms.shutdownEngine()
'''


def hms_command(script):
    """Replicates HEC-HMS.cmd (version 4.10) with properly separated arguments."""
    d = HMS_DIR
    java = d / "jre" / "bin" / ("java.exe" if os.name == "nt" else "java")
    jars = os.pathsep.join([str(d / "hms.jar"), str(d / "lib" / "*")])
    libpath = os.pathsep.join([str(d / "bin"), str(d / "bin" / "gdal"), str(d / "bin" / "hdf")])
    cmd = [str(java), "-Dsun.java2d.d3d=false", "-DMapPanel.NoVolatileImage=true",
           "-Dpython.import.site=false", "-Xms32M", f"-Dpython.path={jars}",
           f"-Djava.library.path={libpath}", "-classpath", jars, "hms.Hms",
           "-script", str(script)]
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(d / "bin"), str(d / "bin" / "gdal"),
                                   str(d / "bin" / "taudem"), env.get("PATH", "")])
    env["GDAL_DRIVER_PATH"] = str(d / "bin" / "gdal" / "gdalplugins")
    env["PROJ_LIB"] = str(d / "bin" / "gdal" / "projlib")
    env["GDAL_DATA"] = str(d / "bin" / "gdal" / "gdal-data")
    return cmd, env, java


def run(scenario: str) -> bool:
    pdir = project_dir(scenario)
    hms_file = pdir / f"{scenario}.hms"
    if not hms_file.exists():
        print(f"[SKIP] {scenario}: {hms_file} not found - run 04_setup_projects.py first")
        return False
    script = pdir / f"compute_{scenario}.py"
    script.write_text(JYTHON.format(hms_file=hms_file, run=RUN_NAME), encoding="utf-8")

    cmd, env, java = hms_command(script)
    if not java.exists():
        print(f"[ERROR] HEC-HMS Java runtime not found: {java}\n"
              f"        Unzip HEC-HMS 4.10 into {HMS_DIR} or edit HMS_DIR in common.py")
        return False

    out_dss = pdir / f"{RUN_NAME}.dss"
    if out_dss.exists():
        out_dss.unlink()                       # make sure results are fresh
    print(f"Running HEC-HMS: {scenario} ...", flush=True)
    res = subprocess.run(cmd, cwd=str(HMS_DIR), env=env, capture_output=True, text=True)
    ok = res.returncode == 0 and out_dss.exists()
    print(f"  {'finished' if ok else 'FAILED'} (return code {res.returncode})")
    if not ok:
        print(res.stdout[-3000:], res.stderr[-3000:], sep="\n")

    log = pdir / f"{RUN_NAME}.log"
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        errors = [ln for ln in lines if ln.startswith("ERROR")]
        warns = [ln for ln in lines if ln.startswith("WARNING")]
        print(f"  log: {len(errors)} errors, {len(warns)} warnings ({log.name})")
        for ln in errors[:10]:
            print("  " + ln)
    return ok


if __name__ == "__main__":
    selected = sys.argv[1:] or list(SCENARIOS)
    results = {s: run(s) for s in selected}
    print("\nSummary:", ", ".join(f"{s}: {'OK' if ok else 'FAILED'}" for s, ok in results.items()))
