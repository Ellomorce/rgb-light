# Large Language Model RGB Test Light Version

## RGB

- An implementation for [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) 

## Quick links

* [Environment](#Environment)
* [Retrieval-Augmented Generation Benchmark](#Retrieval-Augmented)
* [Evaluation](#Evaluation)

### Retrieval-Augmented Generation Benchmark

The data is putted in `data/`

```text
data/
├── en.json
├── en_int.json
├── en_fact.json
├── zh.json
├── zh_int.json
└── zh_fact.json
```

To evalute the Information Integration, you should use `zh_int` or `en_int` for Chinese questions or English questions. 

To evalute the Counterfactual Robustness, you should use `zh_fact` or `en_fact` for Chinese questions or English questions. 


### Evaluation

`temp` is the temperature of model.

`noise_rate` is rate of noisy documents in inputs.

The outputs are:
+ all_rate: The accuracy (noise_rate<1) or rejection rate (noise_rate=1)
+ fact_check_rate: the error detection rates (ED)

The "reject_rate" in the outputs are the reject rate (Rej\*).

The "reject_rate" in the outputs are the error detection rates (ED\*). The `correct_rate` in the outputs are the error correction rate (CR)

---

Prompt Option

+ "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ’I can not answer the question because of the insufficient information in documents.‘. If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer. Finally, You need to answer all the questions in Traditional Chinese."
+ "你是一個準確可靠的人工智能助手，能夠借助外部文件回答問題，請注意外部文件可能存在雜訊及錯誤事實。如果文件中的資訊包含正確答案，你將進行準確的回答。如果文件中的資訊不包含正確答案，你將生成“文件資訊不足，因此我無法基於提供的文件回答該問題。”。如果部分文件中存在與事實不一致的錯誤，請先生成“文件存在錯誤事實。”，並生成正確答案。"
