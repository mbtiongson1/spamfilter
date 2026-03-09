import os
import subprocess

# source = userdata.get('SourcePath') # for local file, use below:
source = 'trec06p-ai201'

def show_source():
    result = subprocess.run(
        f'ls -R "{source}" | head -n 10',
        shell=True, capture_output=True, text=True
    )
    print(result.stdout)
    print("...")
