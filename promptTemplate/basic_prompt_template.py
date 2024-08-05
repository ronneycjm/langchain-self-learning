from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

prompt_template_init = PromptTemplate(
    template="What is the capital of {country}?", input_variables=["country"]
)
prompt_value_init = prompt_template_init.format(country="France")
print(prompt_value_init)
print("*************************")

# Use a string to create a prompt template, recommended for simple templates
string_template = "What is the capital of {country}?"
prompt_template = PromptTemplate.from_template(string_template)
prompt_value = prompt_template.format(country="China")
print(prompt_value)
print("*************************")

# Use a ChatPromptTemplate to create a prompt template with multiple messages
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Hi, how are you?"),
        ("ai", "I'm good, you can call me {name}. What can I do for you?"),
        ("human", "What is the capital of {country}?"),
    ]
)
chat_prompt_value = chat_prompt_template.invoke({"country": "France", "name": "Alice"})
print(chat_prompt_value, type(chat_prompt_value))
print("*************************")


# Placeholder
prompt_template_placeholder = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{content}"),
        ("ai", "Sorry, can you repeat that? "),
    ]
)

prompt_template_placeholder_value = prompt_template_placeholder.invoke(
    {
        "content": [
            ("human", "Hi, how are you?"),
            ("ai", "I'm good, you can call me Bob. What can I do for you?"),
            ("human", "What is the capital of france?"),
        ]
    }
)

print(prompt_template_placeholder_value, type(prompt_template_placeholder_value))
