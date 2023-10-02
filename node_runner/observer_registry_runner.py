#!/usr/bin/env python
import threading
import time
import rpyc
from rpyc import ThreadedServer
import atexit
from observer_rpc_node import ObserverService

def main():
    reg = rpyc.utils.registry.UDPRegistryServer(allow_listing=True)
    obs = ThreadedServer(ObserverService(), auto_register=True)
    threads = [
        threading.Thread(target=reg.start),
        threading.Thread(target=obs.start)]
    for t in threads:
        t.daemon = True # this lets us kill with KeyboardInterrupt
        t.start()
    atexit.register(reg.close)
    atexit.register(obs.close)
    while True:
        print(f"Active Node Types: {rpyc.list_services()}")
        if len(rpyc.list_services()) > 0:
            for i in rpyc.list_services():
                print(f"Active Nodes of {i}: {rpyc.discover(i)}")
        time.sleep(5)

if __name__ == "__main__":
    main()
