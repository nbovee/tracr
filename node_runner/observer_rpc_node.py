import rpyc
# from rpyc.utils.registry import RegistryServer


class ObserverService(rpyc.Service):

    def exposed_inference_completed_signal(self, uuid):
        # to be implemented by Steve
        pass