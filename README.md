# RAGwithLlamaProject

<p align="center">
  <img src="https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/images/llamas-alpaca.gif" alt="Alt text">
</p>

[credit of picture](http://artpictures.club/autumn-2023.html)

## Motivation
As most people know, LLMs usually give us hallucination responses, and RAG (Retrieval Augmented Generation) is one of the methods that most people use for tackling hallucination problems. This inspires me to explore how RAG techniques can improve LLMs' responses.

## Resource
- **Full Raw Scopus Dataset:** Resource from 2110531 Data Science and Data Engineering Tools, semester 1/2023, Chulalongkorn University, with the support of the CU Office of Academic Resources (2018 - 2023)
- **Embedding Model:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **LLM:** Meta's [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and its tokenizer
- **Langchain Framework**
- **Vector Database:** [Pinecone](https://www.pinecone.io) 



## Methodology
- can find [my source code](https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/LLAMA_2_RAG.ipynb) here 
- **Below is overall steps of my RAG process**
<p align="center">
  <img src="https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/images/connect.jpg" alt="Alt text">
</p>

**For elaborate on steps, I create images to show each step as following below:**

1. Preprocessing Data, choose only some features and clean the data for some null values and undesire things

2. Initializing the HuggingFace Embedding Pipeline, use the embedding pipeline to build our embeddings and store them in a Pinecone vector index, embed and index our prepared data
![Alt text](https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/images/3.jpg)

3. Initializing the text-generation pipeline which require LLMs and its respective tokenizer
![Alt text](https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/images/5.jpg)

4. Utilizing RetrivalQA or RetrivalQAWithSourceChain from Langchain framework to chain LLMs, Pinecone index, Vectorstore to create RAG pipeline
![Alt text](https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/images/6.jpg)

5. Combine all things to make responses to users
   - **Retrieve** context from vectorstore with semantic search
   - **Augment** context and prompt so that feed prompt with related knowledge to LLMs
   - **Generate** response out to users
  
![Alt text](https://github.com/RadchaneepornC/RAGwithLLamaProject/blob/main/images/8.jpg)
   


## Result


## Analysis and further improvement

## Conclusion

## References
- Tutorial: [James Briggs's Better Llama 2 with Retrieval Augmented Generation (RAG)](https://www.youtube.com/watch?v=ypzmPwLH_Q4)
