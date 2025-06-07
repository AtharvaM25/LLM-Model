# A Domain-Specific (Harry Potter Series) Chatbot
This project functions like a chatbot that gives vague or generic replies based on the data it was trained on.

# Overview
This project involves training a Large Language Model (LLM) on a domain-specific dataset, such as the Harry Potter series. The LLM processes user prompts, identifies key terms, and generates contextually relevant responses based on the trained data.

* The dataset is first tokenized  
* Batches are created from the dataset for training  
* These batches are used to train a basic transformer model  
* The trained model is saved to a separate file  
* Once a prompt is given, the model predicts the next word and delivers a response  

# Project Requirements 
* **PyTorch** – Used to build, train, and deploy neural networks efficiently, with GPU support  
* **Jupyter Notebook / Google Colab**  
* **Python**
