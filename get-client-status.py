#!/usr/bin/python3

import Pyro5.api
import sys
import datetime
import re

# from https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def main():
	url = sys.argv[1]

	stuck_interval = datetime.timedelta(minutes=10)

	sys.excepthook = Pyro5.errors.excepthook
	manager = Pyro5.api.Proxy(url)

	reports = manager.get_client_reports()
	besttime = datetime.datetime.utcnow()
	if len(reports) > 0:
		besttime = datetime.datetime.fromisoformat(list(reports.items())[0][1]['time'])

	speed = 0
	bps = 0

	blocks_left = sum(manager.get_blocks_left())

	known_hosts = [
		'mojito.local',
		'JohnCollins',
		'spades1',
		'spades2',
		'spades3',
		'spades4',
		'spades6',
		'spades7',
		'spades8',
		'spades9',
		'spades10',
		'spades11',
		'spades12',
		'spades13',
	]

	for key in reports:
		data = reports[key]
		if datetime.datetime.fromisoformat(data['time']) > besttime:
			besttime = datetime.datetime.fromisoformat(data['time'])

		if data['speed'] > 0:
			speed += data['cores'] / data['speed']
		
		delta = datetime.datetime.utcnow() - datetime.datetime.fromisoformat(data['start']) - datetime.timedelta(seconds=abs(data['pause']))
		bps += data['count'] / delta.total_seconds()

	
	for host in known_hosts:
		if host not in reports:
			reports[host] = {'time': str(besttime - 2 * stuck_interval), 'speed': 0, 'count': 0, 'cores': 0}

	generation_eta = 'inf'
	if bps > 0:
		generation_eta = datetime.timedelta(seconds = int(blocks_left / bps))

	print(f'Computing at {int(speed):0d} speed, {bps:.01f} b/s => {generation_eta}')
	print(f"{'Host':<16s}\tSpeed\tDone\t{'Time':<26s}\tStuck")
	for key in natural_sort(reports):
		data = reports[key]
		speed = 0
		if data['speed'] > 0:
			speed = int(data['cores'] / data['speed'])
		delta = besttime - datetime.datetime.fromisoformat(data['time'])

		stuck = '-'
		if delta > stuck_interval:
			stuck = 'Stuck'

		print(f"{key:<16s}\t{str(speed):<5s}\t{data['count']}\t{data['time']}\t{stuck}")
	
if __name__=="__main__":
	main()
