import subprocess
import tempfile

import yaml

p = subprocess.run('conda env export --from-history'.split(), capture_output=True)
from_history = yaml.safe_load(p.stdout)
dependencies = ['python'] + from_history['dependencies']
for x in ['ca-certificates', 'certifi', 'openssl']:
    dependencies.remove(x)

p = subprocess.run('conda env export --no-builds'.split(), capture_output=True)
no_builds = yaml.safe_load(p.stdout)
no_builds['dependencies'] = [x for x in no_builds['dependencies'] if x.split('=')[0] in dependencies]
del no_builds['prefix']

with open('environment.yml', 'w') as f:
    yaml.dump(no_builds, f)
