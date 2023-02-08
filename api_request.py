from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

app = FastAPI()

@app.get("/")

class item2(BaseModel):
    std: int
    mean: int


class item(BaseModel):
    Model: str
    HT: List[item2]
    PPT: List[item2]
    RRT: List[item2]
    RPT: List[item2]


async def prediction(item:item):
    return item