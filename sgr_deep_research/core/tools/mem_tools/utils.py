import os

from sgr_deep_research.core.tools.mem_tools.settings import (
    FILE_SIZE_LIMIT,
    DIR_SIZE_LIMIT,
    MEMORY_SIZE_LIMIT,
)


def check_file_size_limit(file_path: str) -> bool:
    """
    Check if the file size limit is respected.
    """
    return os.path.getsize(file_path) <= FILE_SIZE_LIMIT

def check_dir_size_limit(dir_path: str) -> bool:
    """
    Check if the directory size limit is respected.
    """
    return os.path.getsize(dir_path) <= DIR_SIZE_LIMIT

def check_memory_size_limit() -> bool:
    """
    Check if the memory size limit is respected.
    """
    current_working_dir = os.getcwd()
    return os.path.getsize(current_working_dir) <= MEMORY_SIZE_LIMIT

def check_size_limits(file_or_dir_path: str) -> bool:
    """
    Check if the size limits are respected.
    """
    if file_or_dir_path == "":
        return check_memory_size_limit()
    elif os.path.isdir(file_or_dir_path):
        return check_dir_size_limit(file_or_dir_path) and check_memory_size_limit()
    elif os.path.isfile(file_or_dir_path):
        parent_dir = os.path.dirname(file_or_dir_path)
        if not parent_dir == "":
            return (
                check_file_size_limit(file_or_dir_path)
                and check_dir_size_limit(parent_dir)
                and check_memory_size_limit()
            )
        else:
            return check_file_size_limit(file_or_dir_path) and check_memory_size_limit()
    else:
        return False