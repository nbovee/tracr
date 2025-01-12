# src/api/power_monitor.py

import time
import logging
from typing import Union, Optional, Dict, Any, List, Tuple
from pathlib import Path

import psutil  # type: ignore
import torch
import pandas as pd
import platform
import subprocess

logger = logging.getLogger("split_computing_logger")