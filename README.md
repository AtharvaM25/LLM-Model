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

# To run the program:-
1.  **Ensure you have the necessary environment setup.** This includes Python installed on your system and the required libraries (PyTorch, KaggleHub). You can install them using pip:
    ```bash
    pip install torch kagglehub
    ```
2.  **Download the pre-trained model file.** The `bot.py` script expects a file named `model-01.pkl` to be present in the same directory as the script. This file contains the trained GPT model. You will need to obtain this file separately (e.g., from a training script or a provided download link).
3.  **Place `bot.py` and `model-01.pkl` in the same directory.**
4.  **Execute the script from your terminal:**
    ```bash
    python bot.py
    ```
5.  **Enter your prompt.** Once the script starts, it will prompt you with "Prompt:\n". Type your desired text and press Enter. The model will then generate a completion based on your input.
