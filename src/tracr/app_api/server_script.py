SERVER_SCRIPT = r"""\
import sys
import os
import atexit
import shutil
import logging
import logging.handlers
import rpyc.core.protocol
from importlib import import_module
from threading import Event


rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


server_module     = "$SVR-MODULE$"
server_class      = "$SVR-CLASS$"
model_class       = "$MOD-CLASS$"
model_module      = "$MOD-MODULE$"
ps_module         = "$PS-MODULE$"
ps_class          = "$PS-CLASS$"
node_name         = "$NODE-NAME$".upper()
participant_host  = "$PRT-HOST$"
observer_ip       = "$OBS-IP$"
max_uptime        = $MAX-UPTIME$


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
    shutil.rmtree(here, ignore_errors = True)
atexit.register(rmdir)

try:
    for dirpath, _, filenames in os.walk(here):
        for fn in filenames:
            if fn == "__pycache__" or (fn.endswith(".pyc") and os.path.exists(fn[:-1])):
                os.remove(os.path.join(dirpath, fn))
except Exception:
    pass

sys.path.insert(0, here)

logger.info(f"Importing {server_class} from {server_module} as ServerCls'")
m = import_module(server_module)
ServerCls = getattr(m, server_class)

if model_class and model_module:
    logger.info(f"Importing {model_class} from experiment_design.models.{model_module}.")
    m = import_module(f"experiment_design.models.{model_module}")
    Model = getattr(m, model_class)
else:
    logger.info("Using default model (AlexNet)")
    Model = None

logger.info(f"Importing {ps_class} from experiment_design.services.{ps_module}.")
m = import_module(f"experiment_design.services.{ps_module}")
CustomParticipantService = getattr(m, ps_class)

# One way to programmatically set the service's formal name
class $NODE-NAME$Service(CustomParticipantService):
    ALIASES = [node_name, "PARTICIPANT"]

logger.info("Constructing participant_service instance.")
participant_service = $NODE-NAME$Service(Model)

done_event = Event()
participant_service.link_done_event(done_event)

logger.info("Starting RPC server in thread.")
server = ServerCls(participant_service,
                   port=18861,
                   reuse_addr=True,
                   logger=logger,
                   auto_register=True,
                   protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)

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

try:
    done_event.wait(timeout=max_uptime)
finally:
    close_server_finally()
"""
