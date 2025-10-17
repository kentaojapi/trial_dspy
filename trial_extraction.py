# https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
import os
from pprint import pprint

import dspy


api_key = os.environ["OPENAI_API_KEY"]
lm = dspy.LM("openai/gpt-4.1-nano", temperature=1, api_key=api_key)
dspy.configure(lm=lm)


import requests
import dspy
import json
import random


def init_dataset():
    # Load from the url
    url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"
    dataset = json.loads(requests.get(url).text)
    dspy_dataset = [
        dspy.Example({
            "message": d['fields']['input'],
            "answer": d['answer'],
        }).with_inputs("message")
        for d in dataset
    ]
    random.Random(0).shuffle(dspy_dataset)
    train_set = dspy_dataset[:int(len(dspy_dataset) * 0.33)]
    val_set = dspy_dataset[int(len(dspy_dataset) * 0.33):int(len(dspy_dataset) * 0.66)]
    test_set = dspy_dataset[int(len(dspy_dataset) * 0.66):]

    return train_set, val_set, test_set


train_set, val_set, test_set = init_dataset()

print("Input Message:")
print(train_set[0]['message'])

print("\n\nGold Answer:")
for k, v in json.loads(train_set[0]['answer']).items():
    print(f"{k}: {v}")


from typing import List, Literal


class FacilitySupportAnalyzerUrgency(dspy.Signature):
    """
    Read the provided message and determine the urgency.
    """
    message: str = dspy.InputField()
    urgency: Literal['low', 'medium', 'high'] = dspy.OutputField()


class FacilitySupportAnalyzerSentiment(dspy.Signature):
    """
    Read the provided message and determine the sentiment.
    """
    message: str = dspy.InputField()
    sentiment: Literal['positive', 'neutral', 'negative'] = dspy.OutputField()


class FacilitySupportAnalyzerCategories(dspy.Signature):
    """
    Read the provided message and determine the set of categories applicable to the message.
    """
    message: str = dspy.InputField()
    categories: List[
        Literal[
            "emergency_repair_services",
            "routine_maintenance_requests",
            "quality_and_safety_concerns",
            "specialized_cleaning_services",
            "general_inquiries",
            "sustainability_and_environmental_practices",
            "training_and_support_requests",
            "cleaning_services_scheduling",
            "customer_feedback_and_complaints",
            "facility_management_issues"
        ]
    ] = dspy.OutputField()


class FacilitySupportAnalyzerMM(dspy.Module):
    def __init__(self):
        self.urgency_module = dspy.ChainOfThought(FacilitySupportAnalyzerUrgency)
        self.sentiment_module = dspy.ChainOfThought(FacilitySupportAnalyzerSentiment)
        self.categories_module = dspy.ChainOfThought(FacilitySupportAnalyzerCategories)
    
    def forward(self, message: str):
        urgency = self.urgency_module(message=message)
        sentiment = self.sentiment_module(message=message)
        categories = self.categories_module(message=message)

        return dspy.Prediction(
            urgency=urgency.urgency,
            sentiment=sentiment.sentiment,
            categories=categories.categories
        )


program = FacilitySupportAnalyzerMM()
