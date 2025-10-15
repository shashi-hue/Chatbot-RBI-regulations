# RAG Chatbot Evaluation with LangSmith & Gemini

This project demonstrates how to evaluate a **Retrieval-Augmented Generation (RAG)** chatbot using **LangSmith** and **Gemini API**.

---

## 🧠 What It Does
- Loads a small dataset of questions and reference answers.  
- Uses a RAG-based chatbot (`get_answer()`) powered by **Gemini** to generate responses.  
- Compares generated answers with reference outputs using an evaluator (like `load_evaluator("qa")`).  
- Logs all results and metrics to **LangSmith** for easy visualization and tracking.

---

## ⚙️ Tools Used
- **LangSmith** – for tracking, dataset creation, and evaluation logs.  
- **Gemini API** – for generating chatbot responses.  
- **Python** – for evaluation scripting.

---

## 📊 Output
- All test examples and evaluation scores appear in the **LangSmith Dashboard**.  
- Each record shows:
  - Question  
  - Chatbot answer  
  - Reference answer  
  - Evaluator feedback (score/reason)

---

## 🚀 How to Run
1. Add your API keys as environment variables:
   ```bash
   export LANGCHAIN_API_KEY="your_langsmith_api_key"
   export GOOGLE_API_KEY="your_gemini_api_key"

   python rag_chatbot.py #for chatbot
   python eval_rag.py  #for eval

   ```

