# A domain specific (Harry-Potter series) chatbot
This project works like a chatbot that gives vague or generic replies based on the data it was trained on.

# Overview
This project involves training a Large Language Model (LLM) on a domain-specific dataset, such as the Harry Potter series. The LLM processes user prompts, identifies key terms, and generates contextually relevant responses based on the trained data.

* Dataset is first tokenized
* Batches are made of the dataset for training
* These batches are used to train a basic transformer model
* The trained data is stored in a separate file
* Once a prompt is given, the model predicts the next word and delivers an answer

#
