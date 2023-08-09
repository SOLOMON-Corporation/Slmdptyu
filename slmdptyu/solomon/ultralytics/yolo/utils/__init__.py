import re
import os
import yaml
import uuid
import urllib
import tempfile
import platform
from pathlib import Path
from typing import Union
from types import SimpleNamespace
from slmdptyu.solomon.ultralytics import __version__
from slmutils.general import LOGGER

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
RANK = int(os.getenv('RANK', -1))
DEFAULT_CFG_PATH = ROOT / 'yolo/cfg/default.yaml'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


class IterableSimpleNamespace(SimpleNamespace):
    """
    Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml
            """)

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def emojis(string=''):
    # Return platform-dependent emoji-safe version of string
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path) or (None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d
    return None  # no .git dir found


def is_git_dir():
    """
    Determines whether the current file is part of a git repository.
    If the current file is not part of a git repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return get_git_dir() is not None


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    try:
        with tempfile.TemporaryFile(dir=dir_path):
            pass
        return True
    except OSError:
        return False


def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        Path: The path to the user config directory.
    """
    # Return the appropriate config directory for each operating system
    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(str(path.parent)):
        path = Path('/tmp') / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = Path(os.getenv('YOLO_CONFIG_DIR', get_user_config_dir()))
SETTINGS_YAML = USER_CONFIG_DIR / 'settings.yaml'


def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.

    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump data to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v
                        for k, v in data.items()},
                       f,
                       sort_keys=False,
                       allow_unicode=True)


def get_settings(file=SETTINGS_YAML, version='0.0.3'):
    """
    Loads a global Ultralytics settings YAML file or creates one with default values if it does not exist.

    Args:
        file (Path): Path to the Ultralytics settings YAML file. Defaults to 'settings.yaml' in the USER_CONFIG_DIR.
        version (str): Settings version. If min settings version not met, new default settings will be saved.

    Returns:
        dict: Dictionary of settings key-value pairs.
    """
    import hashlib

    from slmdptyu.ultralytics.ultralytics.yolo.utils.checks import check_version
    from slmdptyu.ultralytics.ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first

    git_dir = get_git_dir()
    root = git_dir or Path()
    datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()
    defaults = {
        'datasets_dir': str(datasets_root / 'datasets'),  # default datasets directory.
        'weights_dir': str(root / 'weights'),  # default weights directory.
        'runs_dir': str(root / 'runs'),  # default runs directory.
        'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # anonymized uuid hash
        'sync': True,  # sync analytics to help with YOLO development
        'api_key': '',  # Ultralytics HUB API key (https://hub.ultralytics.com/)
        'settings_version': version}  # Ultralytics settings version

    with torch_distributed_zero_first(RANK):
        if not file.exists():
            yaml_save(file, defaults)
        settings = yaml_load(file)

        # Check that settings keys and types match defaults
        correct = \
            settings \
            and settings.keys() == defaults.keys() \
            and all(type(a) == type(b) for a, b in zip(settings.values(), defaults.values())) \
            and check_version(settings['settings_version'], version)
        if not correct:
            LOGGER.warning('WARNING ⚠️ Ultralytics settings reset to defaults. This is normal and may be due to a '
                           'recent ultralytics package update, but may have overwritten previous settings. '
                           f"\nView and update settings with 'yolo settings' or at '{file}'")
            settings = defaults  # merge **defaults with **settings (prefer **settings)
            yaml_save(file, settings)  # save updated defaults

        return settings


def deprecation_warn(arg, new_arg, version=None):
    if not version:
        version = float(__version__[:3]) + 0.2  # deprecate after 2nd major release
    LOGGER.warning(f"WARNING ⚠️ '{arg}' is deprecated and will be removed in 'ultralytics {version}' in the future. "
                   f"Please use '{new_arg}' instead.")


def clean_url(url):
    # Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return urllib.parse.unquote(url).split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
SETTINGS = get_settings()
DATASETS_DIR = Path(SETTINGS['datasets_dir'])