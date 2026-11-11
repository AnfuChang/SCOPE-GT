"""Build/check HubGT's original index extension; no training or dataset access."""
import argparse
import subprocess
import sys
from SCOPE_GT.external_backbones import baseline_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force',action='store_true')
    args = p.parse_args()
    folder = baseline_path('HubGT/Precompute').parent
    command = [sys.executable,'-c',
        'import sys; sys.path[:0]=[".","Precompute"]; import hubgt_npz; hubgt_npz.check_extension()']
    check = subprocess.run(command,cwd=folder)
    if check.returncode == 0 and not args.force: return
    if sys.platform == 'win32':
        raise SystemExit('Build the original OpenMP/Cython extension in the Linux server environment.')
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace','--force'],cwd=folder/'Precompute',check=True)
    subprocess.run(command,cwd=folder,check=True)


if __name__ == '__main__': main()
