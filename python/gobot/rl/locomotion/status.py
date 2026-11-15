"""Read actual process identity and heartbeat; never infer running from a checkpoint."""
import argparse
import json
import os
from pathlib import Path
import socket
import time


def inspect(directory):
    directory=Path(directory)
    path=directory/'status.json'
    if not path.exists():return {'directory':str(directory),'state':'not_started'}
    status=json.loads(path.read_text())
    if status['state'] in {'starting','running','evaluating'}:
        age=time.time()-status.get('heartbeat_unix',0.)
        same_namespace=status.get('pid_namespace')==os.readlink('/proc/self/ns/pid')
        if status.get('host')==socket.gethostname() and same_namespace:
            try:
                stat=Path(f'/proc/{int(status["pid"])}/stat').read_text().split()
                live=stat[21]==str(status.get('process_start_ticks')) and stat[2]!='Z'
            except (OSError,ValueError):live=False
            if not live:status['state']='lost'
            elif age>90:status['state']='unresponsive'
        else:
            status['process_check']='different_host_or_pid_namespace'
            if age>90:status['state']='status_unknown'
        status['heartbeat_age_s']=round(age,1)
    status['directory']=str(directory)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    directories=[a.directory] if (a.directory/'status.json').exists() else sorted(p.parent for p in a.directory.glob('**/status.json'))
    for directory in directories:
        value=inspect(directory)
        print(json.dumps({k:v for k,v in value.items() if k not in {'traceback','latest_metrics','experiment'}},ensure_ascii=False))

if __name__=='__main__':main()
