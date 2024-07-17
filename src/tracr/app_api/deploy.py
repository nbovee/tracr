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

from src.tracr.app_api import device_mgmt as dm
from src.tracr.app_api import utils
from src.tracr.app_api.server_script import SERVER_SCRIPT

logger = logging.getLogger("tracr_logger")


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

        src_root = local.path(utils.get_repo_root() / "src" / "tracr")
        copy(src_root, tmp / "src" / "tracr")

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
