from sgr_deep_research.core.base_tool import BaseTool
from typing import TYPE_CHECKING
from pydantic import Field
import os

if TYPE_CHECKING:
    from sgr_deep_research.core.models import ResearchContext

class GetSizeTool(BaseTool):
    """
    Get the size of a file or directory.

    Args:
        file_or_dir_path: The path to the file or directory. 
                          If empty string, returns total memory directory size.

    Returns:
        The size of the file or directory in bytes.
    """
    reasoning: str = Field(description="Why do you need get size? (1-2 sentences MAX)", max_length=200)
    file_or_dir_path: str = Field(description="The path to the file or directory.")
    
    async def __call__(self, context: ResearchContext) -> str:
        # Handle empty string by returning total memory size
        if not self.file_or_dir_path or self.file_or_dir_path == "":
            # Get the current working directory (which should be the memory root)
            cwd = os.getcwd()
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(cwd):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass
            return total_size
        
        # Otherwise check the specific path
        if os.path.isfile(self.file_or_dir_path):
            return os.path.getsize(self.file_or_dir_path)
        elif os.path.isdir(self.file_or_dir_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.file_or_dir_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass
            return total_size
        else:
            raise FileNotFoundError(f"Path not found: {self.file_or_dir_path}")


    
