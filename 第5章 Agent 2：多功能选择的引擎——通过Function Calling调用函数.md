
## 第5章 Agent 2：多功能选择的引擎——通过Function Calling调用函数

在本章中，我们将深入探讨OpenAI的Function Calling功能，这是一个强大的工具，能够让AI模型更智能地选择和使用各种函数。我们将通过实际例子来展示如何使用Function Calling来创建一个多功能的AI助手。

### 5.1 OpenAI中的Functions

Function Calling是OpenAI API的一个重要特性，它允许语言模型识别何时应该调用预定义的函数，并生成符合函数参数要求的JSON对象。这个功能大大增强了AI模型的实用性，使其能够更好地与外部系统和数据源交互。

#### 5.1.1 什么是Functions

在OpenAI API中，Functions是一种结构化的方式来描述可以被AI模型调用的外部函数。每个Function定义包括：

1. 名称（name）：函数的唯一标识符
2. 描述（description）：函数功能的简短说明
3. 参数（parameters）：函数接受的参数，使用JSON Schema定义

例如，一个获取天气信息的函数可能如下定义：

```python
weather_function = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
    }
}
```

#### 5.1.2 Function的说明文字很重要

在定义Function时，提供清晰、准确的描述对于模型正确理解和使用该函数至关重要。良好的描述应该：

1. 简明扼要地说明函数的用途
2. 指出任何重要的使用注意事项
3. 如果适用，提供使用示例

例如：

```python
{
    "name": "analyze_sentiment",
    "description": "Analyze the sentiment of a given text. Returns a sentiment score between -1 (very negative) and 1 (very positive). Use this for short texts (up to 500 words) in English.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze, up to 500 words"
            }
        },
        "required": ["text"]
    }
}
```

#### 5.1.3 Function定义中的Sample是什么

在Function定义中，可以包含一个`examples`字段来提供函数使用的示例。这有助于模型更好地理解函数的预期输入和输出。例如：

```python
{
    "name": "get_book_recommendation",
    "description": "Get a book recommendation based on genre and preferred length",
    "parameters": {
        "type": "object",
        "properties": {
            "genre": {
                "type": "string",
                "description": "The preferred book genre"
            },
            "length": {
                "type": "string",
                "enum": ["short", "medium", "long"]
            }
        },
        "required": ["genre"]
    },
    "examples": [
        {
            "input": {"genre": "science fiction", "length": "medium"},
            "output": {"title": "Dune", "author": "Frank Herbert"}
        },
        {
            "input": {"genre": "mystery", "length": "short"},
            "output": {"title": "The Hound of the Baskervilles", "author": "Arthur Conan Doyle"}
        }
    ]
}
```

这些示例可以帮助模型理解函数的预期行为，从而生成更准确的函数调用。

#### 5.1.4 什么是Function Calling

Function Calling是指AI模型能够识别何时需要调用特定函数，并生成适当的函数调用参数。这个过程包括：

1. 模型分析用户输入
2. 确定是否需要调用函数
3. 如果需要，选择合适的函数
4. 生成符合函数参数要求的JSON对象

例如，给定用户输入"What's the weather like in London today?"，模型可能会决定调用`get_current_weather`函数，并生成如下参数：

```json
{
    "location": "London",
    "unit": "celsius"
}
```

Function Calling的优势包括：

1. 增强模型能力：允许模型访问外部数据和功能
2. 结构化输出：生成符合预定格式的数据
3. 任务分解：将复杂任务分解为可管理的步骤
4. 灵活性：可以动态添加或修改可用函数

### 5.2 在Playground中定义Function

OpenAI的Playground提供了一个图形界面来实验Function Calling。以下是在Playground中定义和测试Function的步骤：

1. 访问OpenAI Playground：
   打开 https://platform.openai.com/playground

2. 选择聊天模式：
   在右侧面板中，选择 "Chat" 模式。

3. 添加函数：
   在 "Functions" 部分，点击 "Add function" 按钮。

4. 定义函数：
   在弹出的编辑器中，输入函数定义的JSON。例如：

   ```json
   {
     "name": "get_current_weather",
     "description": "Get the current weather in a given location",
     "parameters": {
       "type": "object",
       "properties": {
         "location": {
           "type": "string",
           "description": "The city and state, e.g. San Francisco, CA"
         },
         "unit": {
           "type": "string",
           "enum": ["celsius", "fahrenheit"]
         }
       },
       "required": ["location"]
     }
   }
   ```

5. 保存函数：
   点击 "Save" 按钮保存函数定义。

6. 测试函数调用：
   在聊天输入框中，输入一个可能触发函数调用的问题，例如：
   "What's the weather like in New York today?"

7. 观察模型响应：
   模型应该识别出需要调用`get_current_weather`函数，并生成相应的函数调用参数。

8. 模拟函数执行：
   在Playground中，你需要手动提供函数的返回结果。点击 "Submit function result" 并输入模拟的天气数据。

9. 查看最终响应：
   模型将基于函数返回的结果生成最终回答。

通过这种方式，你可以在Playground中快速测试和优化你的Function定义，观察模型如何理解和使用这些函数。这对于开发和调试基于Function Calling的应用非常有用。

### 5.3 通过Assistants API实现Function Calling

Assistants API提供了一种更结构化的方式来实现Function Calling。让我们通过一个实际的例子来展示如何使用Assistants API来创建一个能够调用函数的AI助手。

首先，我们需要设置环境并定义一些辅助函数：

```python
import openai
import json
import time

openai.api_key = 'your-api-key-here'

def wait_for_run_completion(thread_id, run_id):
    while True:
        run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status == 'completed':
            return run
        elif run.status == 'failed':
            raise Exception(f"Run failed with error: {run.last_error}")
        time.sleep(1)

def get_latest_assistant_message(thread_id):
    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    for message in messages.data:
        if message.role == 'assistant':
            return message.content[0].text.value
    return None
```

现在，让我们创建一个助手并定义一些函数：

```python
# 创建助手
assistant = openai.beta.assistants.create(
    name="Function Calling Assistant",
    instructions="You are an AI assistant capable of calling various functions to help users with their tasks.",
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                    }
                },
                "required": ["symbol"]
            }
        }
    }]
)

# 创建一个新的线程
thread = openai.beta.threads.create()

# 模拟函数实现
def get_current_weather(location, unit="celsius"):
    # 这里应该是实际的天气API调用
    return f"The current temperature in {location} is 22 degrees {unit}."

def get_stock_price(symbol):
    # 这里应该是实际的股票API调用
    return f"The current stock price of {symbol} is $150.75."

# 函数映射
function_map = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price
}
```

现在，我们可以实现主要的对话循环：

```python
while True:
    user_input = input("You: ")
    ifuser_input.lower() == 'exit':
        break

    # 添加用户消息到线程
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    # 运行助手
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # 等待运行完成
    run = wait_for_run_completion(thread.id, run.id)

    # 检查是否需要调用函数
    if run.required_action:
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name in function_map:
                function_result = function_map[function_name](**function_args)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": function_result
                })

        # 提交函数输出
        openai.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )

        # 等待新的运行完成
        run = wait_for_run_completion(thread.id, run.id)

    # 获取并打印助手的回复
    assistant_message = get_latest_assistant_message(thread.id)
    print("Assistant:", assistant_message)

print("Conversation ended.")
```

这个实现展示了如何使用Assistants API来创建一个能够调用函数的AI助手。主要步骤包括：

1. 创建一个助手，并定义它可以使用的函数。
2. 创建一个对话线程。
3. 在一个循环中处理用户输入。
4. 对于每个用户输入，创建一个新的运行（run）。
5. 检查运行是否需要调用函数。
6. 如果需要，执行相应的函数并提交结果。
7. 获取并显示助手的最终回复。

这种方法的优势包括：

1. 结构化的对话管理：Assistants API自动处理对话历史和上下文。
2. 灵活的函数调用：助手可以根据需要调用多个函数。
3. 异步处理：可以轻松处理需要时间的操作，如API调用。
4. 可扩展性：可以轻松添加新的函数和功能。

### 5.4 通过ChatCompletion API来实现Tool Calls

虽然Assistants API提供了一个高级的接口来处理函数调用，但有时我们可能需要更直接地控制对话流程。在这种情况下，我们可以使用ChatCompletion API来实现类似的功能。以下是一个使用ChatCompletion API实现函数调用的例子：

```python
import openai
import json

openai.api_key = 'your-api-key-here'

# 定义可用的函数
available_functions = {
    "get_current_weather": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    },
    "get_stock_price": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                }
            },
            "required": ["symbol"]
        }
    }
}

# 模拟函数实现
def get_current_weather(location, unit="celsius"):
    return f"The current temperature in {location} is 22 degrees {unit}."

def get_stock_price(symbol):
    return f"The current stock price of {symbol} is $150.75."

# 函数映射
function_map = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price
}

# 对话历史
messages = [{"role": "system", "content": "You are a helpful assistant capable of calling functions to aid users."}]

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    messages.append({"role": "user", "content": user_input})

    # 调用ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        functions=list(available_functions.values()),
        function_call="auto"
    )

    assistant_message = response["choices"][0]["message"]

    if assistant_message.get("function_call"):
        # 模型决定调用函数
        function_name = assistant_message["function_call"]["name"]
        function_args = json.loads(assistant_message["function_call"]["arguments"])

        if function_name in function_map:
            function_response = function_map[function_name](**function_args)
            messages.append({
                "role": "function",
                "name": function_name,
                "content": function_response
            })

            # 再次调用API以获取最终响应
            second_response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=messages
            )
            assistant_message = second_response["choices"][0]["message"]

    messages.append(assistant_message)
    print("Assistant:", assistant_message["content"])

print("Conversation ended.")
```

这个实现使用ChatCompletion API来处理对话和函数调用。主要步骤包括：

1. 定义可用的函数及其参数。
2. 维护一个对话历史列表。
3. 在每次用户输入后，调用ChatCompletion API。
4. 如果模型决定调用函数，执行相应的函数并将结果添加到对话历史中。
5. 如果调用了函数，再次调用API以获取最终响应。
6. 将助手的回复添加到对话历史并显示给用户。

使用ChatCompletion API的优势包括：

1. 更直接的控制：可以更精细地控制对话流程和函数调用。
2. 灵活性：可以根据需要动态调整可用的函数和对话策略。
3. 透明度：可以直接看到模型的决策过程，包括是否调用函数以及使用什么参数。
4. 集成性：更容易与现有的基于API的系统集成。

### 5.5 小结

本章我们探讨了OpenAI的Function Calling功能，并通过Assistants API和ChatCompletion API两种方式实现了能够调用函数的AI助手。这种功能大大增强了AI模型的实用性，使其能够更好地与外部系统和数据源交互。

关键点总结：

1. Function Calling允许AI模型识别何时需要调用特定函数，并生成适当的函数调用参数。
2. 在定义函数时，清晰准确的描述和示例非常重要，有助于模型正确理解和使用函数。
3. Assistants API提供了一个高级接口来管理对话和函数调用，适合构建复杂的对话系统。
4. ChatCompletion API允许更直接地控制对话流程和函数调用，适合需要精细控制的场景。
5. 函数调用可以显著扩展AI助手的能力，使其能够访问实时数据、执行计算或与其他系统交互。

实践建议：

1. 仔细设计函数接口，确保它们明确、有用且易于使用。
2. 使用Playground来测试和优化函数定义，观察模型如何理解和使用这些函数。
3. 考虑安全性和错误处理，确保函数调用不会导致意外或不安全的行为。
4. 根据应用需求选择合适的API（Assistants或ChatCompletion），平衡易用性和控制度。
5. 持续监控和改进函数定义和使用，基于实际使用情况进行优化。

通过掌握Function Calling，开发者可以创建更智能、更有用的AI应用，将AI的智能与实际系统的功能无缝结合。这开启了一系列新的可能性，从智能客服到复杂的决策支持系统，Function Calling都可以发挥重要作用。
