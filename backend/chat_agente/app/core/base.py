from pydantic import BaseModel, ConfigDict

class DBModel(BaseModel):
    model_config = ConfigDict(extra='ignore')
