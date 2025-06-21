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
1.  **Ensure you have the necessary environment setup.** This includes Python installed on your system and the required libraries (PyTorch, KaggleHub, Gradio). You can install them using pip:
    ```bash
    pip install torch kagglehub gradio
    ```
2.  **Download the pre-trained model file.** The `bot.py` script expects a file named `model-01.pkl` to be present in the same directory as the script. This file contains the trained GPT model. You will need to obtain this file separately (e.g., from a training script or a provided download link).
3.  **Place `bot.py` and `model-01.pkl` in the same directory.**
4.  **Execute the script from your terminal:**
    ```bash
    python bot.py
    ```
5.  **Open the Chatbot in your browser.** After running the script, Gradio will typically output a local URL (e.g., `http://127.0.0.1:7860`). Copy and paste this URL into your web browser to access the chatbot interface. If `share=True` is enabled in `bot.py`, it will also provide a public shareable link.
6.  **Enter your prompt.** In the Gradio interface, you will see a text box. Type your desired text (e.g., "Harry rushed through the corridor,") and adjust the "Max New Tokens" slider if desired.
7.  **Receive the completion.** Click the "Submit" button (or similar, depending on the Gradio interface design). The model will then process your prompt and generate a continuation of the text, mimicking the style of the Harry Potter books. The generated text will appear in the "Chatbot Response" area.
