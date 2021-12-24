#!/usr/bin/python3

import Pyro5.api
import sys

def main():
	url = sys.argv[1]

	sys.excepthook = Pyro5.errors.excepthook
	manager = Pyro5.api.Proxy(url)

	print(manager.get_client_reports())
	
if __name__=="__main__":
	main()
