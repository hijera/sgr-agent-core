from sgr_deep_research.core.base_tool import BaseTool
from typing import TYPE_CHECKING
from pydantic import Field
import uuid

import os

from sgr_deep_research.core.tools.mem_tools.utils import check_size_limits

if TYPE_CHECKING:
    from sgr_deep_research.core.models import ResearchContext


class CreateFileTool(BaseTool):
    """
    Create a new file in the memory with the given content (if any).
    First create a temporary file with the given content, check if 
    the size limits are respected, if so, move the temporary file to 
    the final destination.

    Args:
        file_path: The path to the file.
        content: The content of the file.

    Returns:
        True if the file was created successfully, False otherwise.
    """

    reasoning: str = Field(description="Why do you need create file? (1-2 sentences MAX)", max_length=200)
    file_path: str = Field(description="The path to the file.")
    content: str = Field(description="The content of the file.")
    
    async def __call__(self, context: ResearchContext) -> str:
        temp_file_path = None
        try:
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(self.file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Create a unique temporary file name in the same directory as the target file
            # This ensures the temp file is within the sandbox's allowed path
            target_dir = os.path.dirname(os.path.abspath(self.file_path)) or "."
            temp_file_path = os.path.join(target_dir, f"temp_{uuid.uuid4().hex[:8]}.txt")
            
            with open(temp_file_path, "w") as f:
                f.write(self.content)
            
            if check_size_limits(temp_file_path):
                # Move the content to the final destination
                with open(self.file_path, "w") as f:
                    f.write(self.content)
                os.remove(temp_file_path)
                return True
            else:
                os.remove(temp_file_path)
                raise Exception(f"File {self.file_path} is too large to create")
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    raise Exception(f"Error removing temp file {temp_file_path}: {e}")
            raise Exception(f"Error creating file {self.file_path}: {e}")


    
