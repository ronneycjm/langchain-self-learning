from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

from langchain_community.chat_models.moonshot import MoonshotChat


# Define your desired data structure.
class CustomStruct(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    intro: str = Field(description="The intro of the person")
    hobbies: list[str] = Field(description="The hobbies of the person")

    @validator("age")
    def age_must_be_positive(cls, value):
        if value < 0:
            raise ValueError("Age must be positive")
        return value


parser = JsonOutputParser(pydantic_object=CustomStruct)


prompt = PromptTemplate(
    template="Give me an intro about {content}.\n{format_instructions}",
    input_variables=["content"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = MoonshotChat()  # Replace with your desired model

chain = prompt | model | parser

print(chain.invoke({"content": "Liu Xiang, a famous Chinese player"}))
