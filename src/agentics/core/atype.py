from pydantic import BaseModel
from typing import Any, Optional

class AGString(BaseModel):
    string:Optional[str] = None


