#!/usr/bin/env python

# test file that creates local nodes for testing
import threading
import time
import rpyc
from rpyc import ThreadedServer
import atexit
import copy
import yaml

from client_rpc_node import ParticipantService
from observer_rpc_node import ObserverServer

participant_servers = []

with open('rpc/test_dicts.yaml', 'r') as file:
    test_dicts = yaml.safe_load(file)
# test_dicts = yaml.load('./test_dicts.yaml', Loader=yaml.Loader)

def create_servers():
    global reg
    reg = ObserverServer(allow_listing=True)
    atexit.register(stop_registry)
    for i in range(5):
        _port = 18861 + i
        stub = ThreadedServer(ParticipantService(i, copy.deepcopy(test_dicts[i])), port=_port, auto_register=True) # since these are all on localhost for dummy testing they must have different ports
        atexit.register(stop_server, i)
        participant_servers.append(stub)

def start_registry():
    print(f"Started Registry Server.")
    reg.start()

def stop_registry():
    print(f"Stopping Registry Server.")
    reg.close()

def start_server(index):
    print(f"Started Node {index} test server.")
    participant_servers[index].start()

def stop_server(index):
    print(f"Stopping Node {index} test server.")
    participant_servers[index].close()

def start_servers():
    threads = [threading.Thread(target=start_registry)]
    for i in range(len(participant_servers)):
        threads.append(threading.Thread(target=start_server, args=[i]))
    for t in threads:
        t.daemon = True # this lets us kill with KeyboardInterrupt
        t.start()

def main():
    create_servers()
    start_servers()
    print(f"Active Node Types:{rpyc.list_services()}")
    print(f"Active Nodes of above: {rpyc.discover(*rpyc.list_services())}")
    # example usage
    if False:
        c1 = rpyc.connect("localhost", 18861)
        ret = c1.root.get_inference('123123') # will be an empty array
        ret1 = c1.root.get_inference('9a6e9b30-2f47-4682-b4b5-a34b67c867fe') # will be an array of matching dicts
    while 1:
        time.sleep(.1)
    


if __name__ == "__main__":
    main()
