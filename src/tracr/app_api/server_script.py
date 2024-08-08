import sys
import os
import atexit
import shutil
import socket
import logging
import logging.handlers
import rpyc.core.protocol
from importlib import import_module
from threading import Event

# Configure RPyC to allow pickling and public attributes
rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True

logger = logging.getLogger("tracr_logger")

# Log the machine's hostname and IP address
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
logger.info(f"Machine hostname: {hostname}")
logger.info(f"Machine IP address: {ip_address}")

# Variables that will be replaced during deployment
server_module = "$SVR-MODULE$"
server_class = "$SVR-CLASS$"
model_class = "$MOD-CLASS$"
model_module = "$MOD-MODULE$"
ps_module = "$PS-MODULE$"
ps_class = "$PS-CLASS$"
node_name = "$NODE-NAME$".upper()
participant_host = "$PRT-HOST$"
observer_ip = "$OBS-IP$"
max_uptime = int("$MAX-UPTIME$")

class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.removesuffix('\n'))
            output = ''.join(self.buf)
            if output:
                self.logfct(output)
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass

def setup_remote_logger(node_name, host, observer_ip):
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.origin = f"{node_name.upper()}@{participant_host}"
        return record

    logging.setLogRecordFactory(record_factory)
    logger = logging.getLogger("tracr_logger")

    logger.setLevel(logging.DEBUG)

    socket_handler = logging.handlers.SocketHandler(observer_ip, 9000)
    logger.addHandler(socket_handler)

    return logger

logger = setup_remote_logger(node_name, participant_host, observer_ip)
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)
logger.info("Zero deploy sequence started.")

logger.info(f"Using Python {str(sys.version_info)}.")
logger.info("Removing __pycache__ and *.pyc files from tempdir.")

here = os.path.dirname(__file__)
os.chdir(here)

def rmdir():
    shutil.rmtree(here, ignore_errors=True)
atexit.register(rmdir)

try:
    for dirpath, _, filenames in os.walk(here):
        for fn in filenames:
            if fn == "__pycache__" or (fn.endswith(".pyc") and os.path.exists(fn[:-1])):
                os.remove(os.path.join(dirpath, fn))
except Exception:
    pass

sys.path.insert(0, here)

logger.info(f"Importing {server_class} from {server_module} as ServerCls")
m = import_module(server_module)
ServerCls = getattr(m, server_class)

if model_class and model_module:
    logger.info(f"Attempting to import {model_class} from {model_module}")
    logger.info(f"sys.path: {sys.path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Contents of current directory: {os.listdir('.')}")
    try:
        m = import_module(model_module)
        Model = getattr(m, model_class)
        logger.info(f"Successfully imported {model_class} from {model_module}")
    except ImportError as e:
        logger.error(f"Failed to import {model_class} from {model_module}: {str(e)}")
        logger.info("Falling back to default model (AlexNet)")
        Model = None
else:
    logger.info("Using default model (AlexNet)")
    Model = None

if Model is None:
    from torchvision.models import alexnet
    Model = alexnet

logger.info(f"Importing {ps_class} from src.tracr.app_api.services.{ps_module}.")
m = import_module(f"src.tracr.app_api.services.{ps_module}")
CustomParticipantService = getattr(m, ps_class)

# One way to programmatically set the service's formal name
ServiceClass = type(f"{node_name}Service", (CustomParticipantService,), {"ALIASES": [node_name, "PARTICIPANT"]})

logger.info("Constructing participant_service instance.")
try:
    participant_service = ServiceClass(Model)
except TypeError:
    logger.warning("ServiceClass does not accept a Model argument. Initializing without Model.")
    participant_service = ServiceClass()

# After creating the participant_service, let's add the model to it if possible
if hasattr(participant_service, 'prepare_model') and callable(getattr(participant_service, 'prepare_model')):
    logger.info("Setting model on participant_service")
    participant_service.prepare_model(Model())  # Create an instance of the Model
elif hasattr(participant_service, 'model') and Model is not None:
    logger.info("Setting model attribute on participant_service")
    participant_service.model = Model()  # Create an instance of the Model
else:
    logger.warning("Unable to set model on participant_service. Service may not function as expected.")

done_event = Event()
participant_service.link_done_event(done_event)

logger.info(f"Starting RPyC server for {node_name} on port 18861")
logger.info(f"Auto-register is set to: True")
logger.info(f"Registry server address: {rpyc.utils.registry.REGISTRY_PORT}")

logger.info("Starting RPC server in thread.")
server = ServerCls(
    participant_service,
    port=18861,
    reuse_addr=True,
    logger=logger,
    auto_register=True,
    protocol_config=rpyc.core.protocol.DEFAULT_CONFIG
)
logger.info(f"RPyC server for {node_name} created successfully")

def close_server_atexit():
    logger.info("Closing server due to atexit invocation.")
    server.close()
    server_thread.join(2)

def close_server_finally():
    logger.info("Closing server after 'finally' clause was reached in SERVER_SCRIPT.")
    server.close()
    server_thread.join(2)

atexit.register(close_server_atexit)

server_thread = server._start_in_thread()
logger.info(f"RPyC server for {node_name} started successfully")

try:
    done_event.wait(timeout=max_uptime)
finally:
    close_server_finally()