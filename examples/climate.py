import asyncio
from typing import List, Optional

from pydantic import BaseModel, Field

from agentics import Agentics as AG


class Location(BaseModel):
    latitude: float = Field(None, description="Decimal(8,6) formating of the latitude")
    longitude: float = Field(
        None, description="Decimal(9,6) formating of the longitude"
    )


class BoundingBox(BaseModel):
    top_left: Location
    top_right: Location
    bottom_left: Location
    bottom_right: Location


class Question(BaseModel):
    question: str
    decomposition: List[str] = Field(
        None,
        description="Decompose the question to ask different sub-questions. We should reason about the different parts that might help you understand the question.",
    )
    context: str = Field(
        None,
        description="Given the question and its decomposed parts, identify additional information that is needed in order to answer the question.",
    )


class Answer(BaseModel):
    result: str


class Region(BaseModel):
    question: Question
    box: BoundingBox
    answer: Optional[Answer] = Field(None)


async def run():
    input = """
    question: "This area is in terra-del-fuago, tell me about the mountain ranges in this area"
    box: "-70.0878451120904060,-54.5019273827569677 : -68.6034773441795380,-53.7124767729584178"
    """

    question = await (AG(atype=Question) << input["question"])
    box = await (AG(atype=BoundingBox) << input["box"])

    region = Region(question=question[0], box=box[0])
    pipeline = AG(atype=Region, states=[region])

    pipeline = await (pipeline("answer", persist=True) << pipeline("question", "box"))

    # # question = "Get the bounding box of the largest of the largest US mountain range."
    # question = "Given the bounding box, tell me where this region is and give me information about the peatlands characteristics in this area"
    # question_agent = AG(atype=Question)
    # question_agent.verbose_transduction = True
    # region_agent = AG(atype=Region)
    # region_agent.verbose_transduction = True

    # pipeline = await (question_agent << [question])
    # pipeline = await (region_agent << pipeline)
    pipeline.pretty_print()


asyncio.run(run())
