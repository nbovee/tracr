import sys
import os
import atexit
import shutil
import time
import logging
import logging.handlers
import rpyc.core.protocol
import traceback
from importlib import import_module
from threading import Event
from pathlib import Path

# Set up basic logging to file and console immediately
log_dir = Path("/tmp/tracr_logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / 'server_script.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

try:
    logging.info("Server script starting...")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Python path: {sys.path}")

    # Configure RPyC to allow pickling and public attributes
    rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
    rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True
    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = 120
    rpyc.core.protocol.DEFAULT_CONFIG["async_request_timeout"] = 120

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

    logging.info(
        f"Configuration: node_name={node_name}, participant_host={participant_host}, observer_ip={observer_ip}")

    class LoggerWriter:
        def __init__(self, logfct):
            self.logfct = logfct
            self.buf = []

        def write(self, msg):
            if msg.endswith("\n"):
                self.buf.append(msg.rstrip())
                output = "".join(self.buf)
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
            record.origin = f"{node_name.upper()}@{host}"
            return record

        logging.setLogRecordFactory(record_factory)
        logger = logging.getLogger("tracr_logger")

        logger.setLevel(logging.DEBUG)

        # Add socket handler for remote logging
        try:
            socket_handler = logging.handlers.SocketHandler(observer_ip, 9000)
            socket_handler.setLevel(logging.DEBUG)
            logger.addHandler(socket_handler)
        except Exception as e:
            logging.error(f"Failed to set up remote logging: {e}")

        return logger

    logger = setup_remote_logger(node_name, participant_host, observer_ip)
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

    logger.info("Zero deploy sequence started.")
    logger.info(f"Using Python {str(sys.version_info)}.")

    here = Path(__file__).parent
    os.chdir(here)

    def rmdir():
        shutil.rmtree(here, ignore_errors=True)

    atexit.register(rmdir)

    try:
        for path in here.rglob("__pycache__"):
            shutil.rmtree(path)
        for path in here.rglob("*.pyc"):
            if path.with_suffix(".py").exists():
                path.unlink()
    except Exception as e:
        logger.error(f"Error cleaning up __pycache__ and .pyc files: {e}")

    sys.path.insert(0, str(here))

    logger.info(f"Importing {server_class} from {server_module} as ServerCls")
    m = import_module(server_module)
    ServerCls = getattr(m, server_class)

    if model_class and model_module:
        logger.info(f"Attempting to import {model_class} from {model_module}")
        logger.info(f"sys.path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Contents of current directory: {os.listdir('.')}")
        m = import_module(model_module)
        Model = getattr(m, model_class)
        logger.info(f"Successfully imported {model_class} from {model_module}")
    else:
        logger.info("Using default model (AlexNet)")
        from torchvision.models import alexnet
        Model = alexnet

    logger.info(
        f"Importing {ps_class} from src.tracr.app_api.services.{ps_module}.")
    m = import_module(f"src.tracr.app_api.services.{ps_module}")
    CustomParticipantService = getattr(m, ps_class)

    ServiceClass = type(
        f"{node_name}Service",
        (CustomParticipantService,),
        {"ALIASES": [node_name, "PARTICIPANT"]},
    )

    logger.info("Constructing participant_service instance.")
    try:
        participant_service = ServiceClass(Model)
    except TypeError:
        logger.warning(
            "ServiceClass does not accept a Model argument. Initializing without Model.")
        participant_service = ServiceClass()

    if hasattr(participant_service, "prepare_model") and callable(getattr(participant_service, "prepare_model")):
        logger.info("Setting model on participant_service")
        model_instance = Model()
        participant_service.prepare_model(model_instance)
    elif hasattr(participant_service, "model") and Model is not None:
        logger.info("Setting model attribute on participant_service")
        participant_service.model = Model()
    else:
        logger.warning(
            "Unable to set model on participant_service. Service may not function as expected.")

    if hasattr(participant_service, "get_connection"):
        try:
            observer_connection = participant_service.get_connection(
                "OBSERVER")
            logger.info("Successfully established connection to OBSERVER")
        except Exception as e:
            logger.error(f"Failed to establish connection to OBSERVER: {e}")
    else:
        logger.warning(
            "ParticipantService does not have a get_connection method. May not be able to access MasterDict.")

    done_event = Event()
    participant_service.link_done_event(done_event)

    def close_server():
        logger.info("Closing server.")
        try:
            server.close()
            logger.info("Server closed successfully.")
        except Exception as e:
            logger.error(f"Error closing server: {e}")

    atexit.register(close_server)

    logger.info(f"Starting RPyC server for {node_name} on port 18861")
    server = ServerCls(
        participant_service,
        port=18861,
        reuse_addr=True,
        logger=logger,
        auto_register=True,
        protocol_config=rpyc.core.protocol.DEFAULT_CONFIG,
    )
    logger.info(f"RPyC server for {node_name} created successfully")

    server_thread = server._start_in_thread()
    logger.info(f"RPyC server for {node_name} started successfully")

    # Wait for the server to register
    for _ in range(30):  # Wait for up to 30 seconds
        if node_name in rpyc.list_services():
            logger.info(f"{node_name} successfully registered")
            break
        time.sleep(1)
    else:
        logger.error(f"{node_name} failed to register within 30 seconds")

    done_event.wait(timeout=max_uptime)

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.error(traceback.format_exc())
    sys.exit(1)

finally:
    logging.info("Server script execution completed.")
    close_server()
