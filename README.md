# Bury College RAG Chatbot with Hybrid Q/A Retrieval

An intelligent, Retrieval-Augmented Generation (RAG) chatbot designed to act as a knowledgeable assistant for Bury College. This system utilises a tiered hybrid retrieval architecture built using Python, FAISS, and the Gemini API, wrapped in a user-friendly Streamlit interface.

## Project Overview
This project was developed as part of the **AGAI-03** curriculum. It automates the end-to-end process of:
* **Web Scraping:** Extracting and cleaning text data from 17 public pages of the Bury College website.
* **Synthetic Q/A Generation:** Leveraging an LLM to generate a high-quality dataset of 204 domain-specific question-and-answer pairs.
* **Dual-Index Vector Store:** Storing embeddings in two separate FAISS indices (one for the synthetic Q/A pairs and one for the raw web chunks).
* **Hybrid Routing Logic:** Employing a distance-based fallback routing system (threshold < 0.5) to deliver highly accurate answers.

---

## Tech Stack & Frameworks
* **Language:** Python (Anaconda / JupyterLab environment)
* **Frontend UI:** Streamlit
* **Vector Database:** FAISS 
* **Embedding & LLM Generation:** Google Gemini API 
* **Web Scraping:** Beautiful Soup / Requests

---

## Local Installation & Setup

* Clone your repository
* Set up your virtual environment
* Install the libraires using pip install -r requirements.txt
* To protect the key create a .env file with this code inside it: GEMINI_API_KEY=your_actual_api_key_here
* The main bulk of the project is on the file called and requires Jupyter Lab to open it: t_rahman_qa_assignment.ipynb
* Run the front end application though your command line interface: streamlit run app.py

## Author
* **Mr Timur Rahman**
