# Large-Language-Model-Application-Development-Hands-On-with-AI-Agents
大模型应用开发：动手做AI Agents
 
陈光剑 编著

<img src="static/ai_genius_institute.jpeg" width="100%" height="auto">

AI 天才研究院 / AI Genius Institute, 2024


# 全书目录

# 大模型应用开发：动手做AI Agents


人工智能时代一种全新的技术——Agent正在崛起。这是一种能够理解自然语言并生成对应回复以及执行具体行动的人工智能体。它不仅是内容生成工具，而且是连接复杂任务的关键纽带。本书将探索Agent的奥秘，内容包括从技术框架到开发工具，从实操项目到前沿进展，通过带着读者动手做7个功能强大的Agent，全方位解析Agent的设计与实现。本书最后展望了Agent的发展前景和未来趋势。

本书适合对Agent技术感兴趣或致力于该领域的研究人员、开发人员、产品经理、企业负责人，以及高等院校相关专业师生等阅读。读者将跟随咖哥和小雪的脚步，踏上饶有趣味的Agent开发之旅，零距离接触GPT-4模型、OpenAI Assistants API、LangChain、LlamaIndex和MetaGPT等尖端技术，见证Agent在办公自动化、智能调度、知识整合以及检索增强生成（RAG）等领域的非凡表现，携手开启人工智能时代的无限可能，在人机协作的星空中共同探寻那颗最闪亮的Agent之星！


前言


第 1 章 何谓Agent，为何Agent 001

[第1章 何谓Agent，为何Agent.md](%E7%AC%AC1%E7%AB%A0%20%E4%BD%95%E8%B0%93Agent%EF%BC%8C%E4%B8%BA%E4%BD%95Agent.md)

1.1 大开脑洞的演讲：Life 3.0 001

1.2 那么，究竟何谓Agent 003

1.3 Agent的大脑：大模型的通用推理能力 006

1.3.1 人类的大脑了不起 006

1.3.2 大模型出现之前的Agent 007

1.3.3 大模型就是Agent的大脑 008

1.3.4 期望顶峰和失望低谷 010

1.3.5 知识、记忆、理解、表达、推理、反思、泛化和自我提升 012

1.3.6 基于大模型的推理能力构筑AI应用 015

1.4 Agent的感知力：语言交互能力和多模态能力 016

1.4.1 语言交互能力 016

1.4.2 多模态能力 016

1.4.3 结合语言交互能力和多模态能力 017

1.5 Agent的行动力：语言输出能力和工具使用能力 017

1.5.1 语言输出能力 017

1.5.2 工具使用能力 018

1.5.3 具身智能的实现 019

1.6 Agent对各行业的效能提升 019

1.6.1 自动办公好助手 020

1.6.2 客户服务革命 020

1.6.3 个性化推荐 020

1.6.4 流程的自动化与资源的优化 021

1.6.5 医疗保健的变革 021

1.7 Agent带来新的商业模式和变革 022

1.7.1 Gartner的8项重要预测 023

1.7.2 Agent即服务 024

1.7.3 多Agent协作 025

1.7.4 自我演进的AI 026

1.7.5 具身智能的发展 026

1.8 小结 027

第 2章 基于大模型的Agent技术框架 029

[第2章 基于大模型的Agent技术框架.md](%E7%AC%AC2%E7%AB%A0%20%E5%9F%BA%E4%BA%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9A%84Agent%E6%8A%80%E6%9C%AF%E6%A1%86%E6%9E%B6.md)

2.1 Agent的四大要素 029

2.2 Agent的规划和决策能力 031

2.3 Agent的各种记忆机制 032

2.4 Agent的核心技能：调用工具 033

2.5 Agent的推理引擎：ReAct框架 035

2.5.1 何谓ReAct 035

2.5.2 用ReAct框架实现简单Agent 038

2.5.3 基于ReAct框架的提示 040

2.5.4 创建大模型实例 043

2.5.5 定义搜索工具 044

2.5.6 构建ReAct Agent 044

2.5.7 执行ReAct Agent 045

2.6 其他Agent认知框架 047

2.6.1 函数调用 047

2.6.2 计划与执行 048

2.6.3 自问自答 048

2.6.4 批判修正 048

2.6.5 思维链 048

2.6.6 思维树 048

2.7 小结 049

第3章 OpenAI API、LangChain和LlamaIndex 051

[第3章 OpenAI API、LangChain和LlamaIndex.md](%E7%AC%AC3%E7%AB%A0%20OpenAI%20API%E3%80%81LangChain%E5%92%8CLlamaIndex.md)

3.1 何谓OpenAI API 052

3.1.1 说说OpenAI这家公司 052

3.1.2 OpenAI API和Agent开发 055

3.1.3 OpenAI API的聊天程序示例 057

3.1.4 OpenAI API的图片生成示例 063

3.1.5 OpenAI API实践 065

3.2 何谓LangChain 067

3.2.1 说说LangChain 068

3.2.2 LangChain中的六大模块 073

3.2.3 LangChain和Agent开发 074

3.2.4 LangSmith的使用方法 075

3.3 何谓LlamaIndex 077

3.3.1 说说LlamaIndex 077

3.3.2 LlamaIndex和基于RAG的AI开发 078

3.3.3 简单的LlamaIndex开发示例 081

3.4 小结 084

第4章 Agent 1：自动化办公的实现——通过Assistants API和DALL·E 3模型创作PPT 085


[第4章 Agent 1：自动化办公的实现——通过Assistants API和DALL·E 3模型创作PPT.md](%E7%AC%AC4%E7%AB%A0%20Agent%201%EF%BC%9A%E8%87%AA%E5%8A%A8%E5%8C%96%E5%8A%9E%E5%85%AC%E7%9A%84%E5%AE%9E%E7%8E%B0%E2%80%94%E2%80%94%E9%80%9A%E8%BF%87Assistants%20API%E5%92%8CDALL%C2%B7E%203%E6%A8%A1%E5%9E%8B%E5%88%9B%E4%BD%9CPPT.md)


4.1 OpenAI公司的Assistants是什么 086

4.2 不写代码，在Playground中玩Assistants 086

4.3 Assistants API的简单示例 090

4.3.1 创建助手 091

4.3.2 创建线程 095

4.3.3 添加消息 097

4.3.4 运行助手 099

4.3.5 显示响应 103

4.4 创建一个简短的虚构PPT 105

4.4.1 数据的收集与整理 106

4.4.2 创建OpenAI助手 106

4.4.3 自主创建数据分析图表 108

4.4.4 自主创建数据洞察 112

4.4.5 自主创建页面标题 114

4.4.6 用DALL·E 3模型为PPT首页配图 115

4.4.7 自主创建PPT 116

4.5 小结 121

第5章 Agent 2：多功能选择的引擎——通过Function Calling调用函数 122

[第5章 Agent 2：多功能选择的引擎——通过Function Calling调用函数.md](%E7%AC%AC5%E7%AB%A0%20Agent%202%EF%BC%9A%E5%A4%9A%E5%8A%9F%E8%83%BD%E9%80%89%E6%8B%A9%E7%9A%84%E5%BC%95%E6%93%8E%E2%80%94%E2%80%94%E9%80%9A%E8%BF%87Function%20Calling%E8%B0%83%E7%94%A8%E5%87%BD%E6%95%B0.md)

5.1 OpenAI中的Functions 122

5.1.1 什么是Functions 123

5.1.2 Function的说明文字很重要 124

5.1.3 Function定义中的Sample是什么 124

5.1.4 什么是Function Calling 126

5.2 在Playground中定义Function 127

5.3 通过Assistants API实现Function Calling 130

5.3.1 创建能使用Function的助手 131

5.3.2 不调用Function，直接运行助手 133

5.3.3 在Run进入requires_action状态之后跳出循环 140

5.3.4 拿到助手返回的元数据信息 141

5.3.5 通过助手的返回信息调用函数 141

5.3.6 通过submit_tool_outputs提交结果以完成任务 143

5.4 通过ChatCompletion API来实现Tool Calls 147

5.4.1 初始化对话和定义可用函数 148

5.4.2 第 一次调用大模型，向模型发送对话及工具定义，并获取响应 149

5.4.3 调用模型选择的工具并构建新消息 151

5.4.4 第二次向大模型发送对话以获取最终响应 153

5.5 小结 154

第6章 Agent 3：推理与行动的协同——通过LangChain中的ReAct框架实现自动定价 156

[第6章 Agent 3：推理与行动的协同——通过LangChain中的ReAct框架实现自动定价.md](%E7%AC%AC6%E7%AB%A0%20Agent%203%EF%BC%9A%E6%8E%A8%E7%90%86%E4%B8%8E%E8%A1%8C%E5%8A%A8%E7%9A%84%E5%8D%8F%E5%90%8C%E2%80%94%E2%80%94%E9%80%9A%E8%BF%87LangChain%E4%B8%AD%E7%9A%84ReAct%E6%A1%86%E6%9E%B6%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%AE%9A%E4%BB%B7.md)

6.1 复习ReAct框架 156

6.2 LangChain中ReAct Agent 的实现 159

6.3 LangChain中的工具和工具包 160

6.4 通过create_react_agent创建鲜花定价Agent 162

6.5 深挖AgentExecutor的运行机制 166

6.5.1 在AgentExecutor中设置断点 166

6.5.2 第 一轮思考：模型决定搜索 169

6.5.3 第 一轮行动：工具执行搜索 175

6.5.4 第二轮思考：模型决定计算 179

6.5.5 第二轮行动：工具执行计算 180

6.5.6 第三轮思考：模型完成任务 182

6.6 小结 185

第7章 Agent 4：计划和执行的解耦——通过LangChain中的Plan-and-Execute实现智能调度库存 186

[第7章 Agent 4：计划和执行的解耦——通过LangChain中的Plan-and-Execute实现智能调度库存.md](%E7%AC%AC7%E7%AB%A0%20Agent%204%EF%BC%9A%E8%AE%A1%E5%88%92%E5%92%8C%E6%89%A7%E8%A1%8C%E7%9A%84%E8%A7%A3%E8%80%A6%E2%80%94%E2%80%94%E9%80%9A%E8%BF%87LangChain%E4%B8%AD%E7%9A%84Plan-and-Execute%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E8%B0%83%E5%BA%A6%E5%BA%93%E5%AD%98.md)

7.1 Plan-and-Solve策略的提出 186

7.2 LangChain中的Plan-and-Execute Agent 190

7.3 通过Plan-and-Execute Agent实现物流管理 192

7.3.1 为Agent定义一系列进行自动库存调度的工具 192

7.3.2 创建Plan-and-Execute Agent并尝试一个“不可能完成的任务” 193

7.3.3 完善请求，让Agent完成任务 200

7.4 从单Agent到多Agent 203

7.5 小结 204

第8章 Agent 5：知识的提取与整合——通过LlamaIndex实现检索增强生成 205

[第8章 Agent 5：知识的提取与整合——通过LlamaIndex实现检索增强生成.md](%E7%AC%AC8%E7%AB%A0%20Agent%205%EF%BC%9A%E7%9F%A5%E8%AF%86%E7%9A%84%E6%8F%90%E5%8F%96%E4%B8%8E%E6%95%B4%E5%90%88%E2%80%94%E2%80%94%E9%80%9A%E8%BF%87LlamaIndex%E5%AE%9E%E7%8E%B0%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90.md)

8.1 何谓检索增强生成 206

8.1.1 提示工程、RAG与微调 206

8.1.2 从技术角度看检索部分的Pipeline 208

8.1.3 从用户角度看RAG流程 209

8.2 RAG和Agent 210

8.3 通过LlamaIndex的ReAct RAG Agent实现花语秘境财报检索 211

8.3.1 获取并加载电商的财报文件 211

8.3.2 将财报文件的数据转换为向量数据 211

8.3.3 构建查询引擎和工具 213

8.3.4 配置文本生成引擎大模型 214

8.3.5 创建 Agent以查询财务信息 214

8.4 小结 215

第9章 Agent 6：GitHub的网红聚落——AutoGPT、BabyAGI和CAMEL 216

[第9章 Agent 6：GitHub的网红聚落——AutoGPT、BabyAGI和CAMEL.md](%E7%AC%AC9%E7%AB%A0%20Agent%206%EF%BC%9AGitHub%E7%9A%84%E7%BD%91%E7%BA%A2%E8%81%9A%E8%90%BD%E2%80%94%E2%80%94AutoGPT%E3%80%81BabyAGI%E5%92%8CCAMEL.md)

9.1 AutoGPT 217

9.1.1 AutoGPT简介 217

9.1.2 AutoGPT实战 218

9.2 BabyAGI 222

9.2.1 BabyAGI简介 222

9.2.2 BabyAGI实战 224

9.3 CAMEL 236

9.3.1 CAMEL简介 236

9.3.2 CAMEL论文中的股票交易场景 237

9.3.3 CAMEL实战 241

9.4 小结 248

第 10章 Agent 7：多Agent框架——AutoGen和MetaGPT 250

[第10章 Agent 7：多Agent框架——AutoGen和MetaGPT.md](%E7%AC%AC10%E7%AB%A0%20Agent%207%EF%BC%9A%E5%A4%9AAgent%E6%A1%86%E6%9E%B6%E2%80%94%E2%80%94AutoGen%E5%92%8CMetaGPT.md)

10.1 AutoGen 250

10.1.1 AutoGen简介 250

10.1.2 AutoGen实战 253

10.2 MetaGPT 256

10.2.1 MetaGPT简介 256

10.2.2 MetaGPT实战 257

10.3 小结 263

附录A 下一代Agent的诞生地：科研论文中的新思路 264

[附录A 下一代Agent的诞生地：科研论文中的新思路.md](%E9%99%84%E5%BD%95A%20%E4%B8%8B%E4%B8%80%E4%BB%A3Agent%E7%9A%84%E8%AF%9E%E7%94%9F%E5%9C%B0%EF%BC%9A%E7%A7%91%E7%A0%94%E8%AE%BA%E6%96%87%E4%B8%AD%E7%9A%84%E6%96%B0%E6%80%9D%E8%B7%AF.md)

A.1 两篇高质量的Agent综述论文 264

A.2 论文选读：Agent自主学习、多Agent合作、Agent可信度的评估、边缘系统部署以及具身智能落地 266

A.3 小结 267

参考文献 269

[参考文献.md](%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AE.md)

后记 创新与变革的交汇点 271

[后记 创新与变革的交汇点.md](%E5%90%8E%E8%AE%B0%20%E5%88%9B%E6%96%B0%E4%B8%8E%E5%8F%98%E9%9D%A9%E7%9A%84%E4%BA%A4%E6%B1%87%E7%82%B9.md)


----

 

# 捐赠：AI天才研究院

> Donate to AI Genius Institute:


| 微信                                                      | 支付宝                                                     |
|---------------------------------------------------------|---------------------------------------------------------|
| <img src="static/wechat.jpeg" width="300" height="350"> | <img src="static/alipay.jpeg" width="300" height="350"> |

