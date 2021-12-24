#!/usr/bin/python3

import Pyro5.api
import sys
import datetime

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

	for key in reports:
		data = reports[key]
		if datetime.datetime.fromisoformat(data['time']) > besttime:
			besttime = datetime.datetime.fromisoformat(data['time'])

		if data['speed'] > 0:
			speed += data['cores'] / data['speed']
	
	print(f'Computing at {int(speed):0d} speed')
	print(f"{'Host':<16s}\tSpeed\tDone\t{'Time':<26s}\tStuck")
	for key in reports:
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
