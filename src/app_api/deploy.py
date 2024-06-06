from plumbum.machines.session import ShellSessionError
import rpyc
import logging
import sys
from rpyc.utils.zerodeploy import DeployedServer, TimeoutExpired
from rpyc.core.stream import SocketStream
from plumbum.machines.remote import RemoteCommand
from plumbum import SshMachine, local, CommandNotFound
from plumbum.path import copy
from plumbum.commands.base import BoundCommand

import src.app_api.device_mgmt as dm
import src.app_api.utils as utils


logger = logging.getLogger("tracr_logger")


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
    logger.info(f"Importing {model_class} from src.experiment_design.models.{model_module}.")
    m = import_module(f"src.experiment_design.models.{model_module}")
    Model = getattr(m, model_class)
else:
    logger.info("Using default model (AlexNet)")
    Model = None

logger.info(f"Importing {ps_class} from src.experiment_design.services.{ps_module}.")
m = import_module(f"src.experiment_design.services.{ps_module}")
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


class ZeroDeployedServer(DeployedServer):

    def __init__(
        self,
        device: dm.Device,
        node_name: str,
        model: tuple[str, str],
        participant_service: tuple[str, str],
        server_class="rpyc.utils.server.ThreadedServer",
        python_executable=None,
        timeout_s: int = 600,
    ):
        logger.debug(f"Constructing ZeroDeployedServer for {node_name}.")
        assert device.working_cparams is not None
        self.name = device._name
        self.proc = None
        self.remote_machine = device.as_pb_sshmachine()
        self._tmpdir_ctx = None

        # Create a temp dir on the remote machine where we make the environment
        self._tmpdir_ctx = self.remote_machine.tempdir()
        tmp = self._tmpdir_ctx.__enter__()

        # Copy over the rpyc and experiment_design packages
        rpyc_root = local.path(rpyc.__file__).up()
        copy(rpyc_root, tmp / "rpyc")

        src_root = local.path(utils.get_repo_root() / "src")
        copy(src_root, tmp / "src")

        # Substitute placeholders in the remote script and send it over
        script = tmp / "deployed-rpyc.py"
        modname, clsname = server_class.rsplit(".", 1)
        m_module, m_class = model
        ps_module, ps_class = participant_service
        observer_ip = utils.get_local_ip()
        participant_host = device.working_cparams.host
        script.write(
            SERVER_SCRIPT.replace("$SVR-MODULE$", modname)  # type: ignore
            .replace("$SVR-CLASS$", clsname)
            .replace("$MOD-MODULE$", m_module)
            .replace("$MOD-CLASS$", m_class)
            .replace("$PS-MODULE$", ps_module)
            .replace("$PS-CLASS$", ps_class)
            .replace("$NODE-NAME$", node_name)
            .replace("$OBS-IP$", observer_ip)
            .replace("$PRT-HOST$", participant_host)
            .replace("$MAX-UPTIME$", str(timeout_s))
        )
        if isinstance(python_executable, BoundCommand):
            cmd = python_executable
        elif python_executable:
            cmd = self.remote_machine[python_executable]
        else:
            major = sys.version_info[0]
            minor = sys.version_info[1]
            logger.info(
                f"Observer uses Python {major}.{minor}. Looking for equivalent Python executable on {node_name}"
            )
            cmd = None
            for opt in [f"python{major}.{minor}", f"python{major}"]:
                try:
                    logger.info(f"Checking {opt}")
                    cmd = self.remote_machine[opt]
                    logger.info(f"{opt} is available.")
                except CommandNotFound:
                    logger.info(f"{opt} is not available.")
                    pass
                else:
                    break
            if not cmd:
                logger.warning(
                    "Had to use the default python interpreter, which could cause problems."
                )
                cmd = self.remote_machine.python

        assert isinstance(cmd, RemoteCommand)
        self.proc = cmd.popen(script, new_session=True)

    def _connect_sock(self):
        assert isinstance(self.remote_machine, SshMachine)
        return SocketStream._connect(self.remote_machine.host, 18861)

    def __del__(self):
        try:
            super().__del__
        except (AttributeError, ShellSessionError):
            pass

    def close(self, timeout=5):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.communicate(timeout=timeout)
            except TimeoutExpired:
                self.proc.kill()
                raise
            except Exception:
                pass
            self.proc = None
        if self.remote_machine is not None:
            try:
                self.remote_machine._session.proc.terminate()
                self.remote_machine._session.proc.communicate(timeout=timeout)
                self.remote_machine.close()
            except TimeoutExpired:
                self.remote_machine._session.proc.kill()
                raise
            except ShellSessionError:
                logger.info(f"remote machine {self.name} has been closed")
            except Exception:
                pass
            self.remote_machine = None
        if self._tmpdir_ctx is not None:
            try:
                self._tmpdir_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._tmpdir_ctx = None
