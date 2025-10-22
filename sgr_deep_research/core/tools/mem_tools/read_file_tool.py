from sgr_deep_research.core.base_tool import BaseTool
from typing import TYPE_CHECKING
from pydantic import Field

import os

if TYPE_CHECKING:
    from sgr_deep_research.core.models import ResearchContext


class ReadFileTool(BaseTool):
    """
    Read a file in the memory.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file, or an error message if the file cannot be read.
    """

    reasoning: str = Field(description="Why do you need read file? (1-2 sentences MAX)", max_length=200)
    file_path: str = Field(description="The path to the file.")
    
    async def __call__(self, context: ResearchContext) -> str:
            try:
                # Ensure the file path is properly resolved
                if not os.path.exists(self.file_path):
                    return f"Error: File {self.file_path} does not exist"
                
                if not os.path.isfile(self.file_path):
                    return f"Error: {self.file_path} is not a file"
                    
                with open(self.file_path, "r") as f:
                    return f.read()
            except PermissionError:
                return f"Error: Permission denied accessing {self.file_path}"
            except Exception as e:
                return f"Error: {e}"


    
