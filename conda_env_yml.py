import subprocess
import tempfile

import yaml

p = subprocess.run("conda env export --from-history".split(), capture_output=True)
from_history = yaml.safe_load(p.stdout)
dependencies = ["python"] + from_history["dependencies"]
for x in ["ca-certificates", "certifi", "openssl"]:
    dependencies.remove(x)

p = subprocess.run("conda env export --no-builds".split(), capture_output=True)
no_builds = yaml.safe_load(p.stdout)


def extract_no_build_deps(deps):
    rv = []
    for dep in deps:
        if isinstance(dep, dict):
            # Skip pip dependencies.
            continue
        else:
            if dep.split("=")[0] in dependencies:
                rv.append(dep)

    if not any(x.startswith("pip") for x in rv):
        rv.append("pip")
    rv.append({"pip": ["git+https://github.com/spenceforce/NLP-Simple-to-Spectacular"]})

    return rv


no_builds["dependencies"] = extract_no_build_deps(no_builds["dependencies"])
del no_builds["prefix"]

with open("environment.yml", "w") as f:
    yaml.dump(no_builds, f)
