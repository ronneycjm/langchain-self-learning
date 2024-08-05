from typing import Any, Dict
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

### The usage of example selector is to select examples from the a large number of examples.
examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivederci"},
    {"input": "soccer", "output": "calcio"},
]


class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, Any]) -> Dict[str, Any]:
        # This assumes knowledge that part of the input will be a 'text' key
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        # Initialize variables to store the best match and its length difference
        best_match = None
        smallest_diff = float("inf")

        # Iterate through each example
        for example in self.examples:
            # Calculate the length difference with the first word of the example
            current_diff = abs(len(example["input"]) - new_word_length)

            # Update the best match if the current one is closer in length
            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]


sample_selector = CustomExampleSelector(examples)
# print(sample_selector.select_examples({"input": "hi"}))


### Few shot template is used to generate a prompt for a given input
example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")
fewshot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=sample_selector,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to Italian:",
    input_variables=["input"],
)
print(fewshot_prompt.invoke({"input": "bye"}))


examples1 = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "What is the capital of Germany?", "output": "Berlin"},
    {"input": "What is the capital of Italy?", "output": "Rome"},
]
example_prompt_1 = PromptTemplate.from_template(
    "InputCountry: {input} -> OutputCaptial: {output}"
)
fewshot_prompt_1 = FewShotPromptTemplate(
    examples=examples1,
    example_prompt=example_prompt_1,
    prefix="Tell me the captial of a country.",
    suffix="InputCountry: {country} -> OutputCaptial:",
    input_variables=["country"],
)


print(fewshot_prompt_1.invoke({"country": "Spain"}))
