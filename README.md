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
| Prompt | Response from Llama 13b chat|  Response from RAG pipeline|
|-------------------|------|--------------|
|Publications about Neural Network and Large Language Models|Here are some recent research papers and articles about neural networks and large language models that you may find interesting:\n\n1.**"Attention is All You Need"** by Ashish Vaswani et al. This paper introduced the Transformer architecture, which relies solely on self-attention mechanisms to process input sequences. The Transformer has since become a popular choice for many NLP tasks.\n2. **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** by Jacob Devlin et al. This paper introduced BERT, a pre-trained language model that achieved state-of-the-art results on a wide range of NLP tasks. BERT uses a multi-layer bidirectional transformer encoder to generate contextualized representations of words in a sentence.\n3. **"Language Models as a Tool for Exploring Human Cognition"** by Christopher Manning and Hinrich Schütze. This paper discusses the potential use of language models as a tool for exploring human cognition and understanding the nature of language.\n4. **"The Unreasonable Effectiveness of Recurrent Neural Networks"** by David Mimno. This paper provides an overview of the success of recurrent neural networks (RNNs) in NLP tasks, and discusses some of the challenges and open questions in the field.\n5. **"A Survey on Transfer Learning for Natural Language Processing"** by Zhiyu Zhang and Qun Liu. This paper provides a comprehensive survey of transfer learning techniques for NLP tasks, including the use of pre-trained language models like BERT.\n6. **"Deep Learning for Natural Language Processing: A Review and Perspective"** by Yue Zhang and Jun Zhu. This paper reviews recent advances in deep learning for NLP, including the use of convolutional neural networks (CNNs), RNNs, and transformer-based architectures.\n7. **"The Impact of Language Models on the Future of Work"** by Martin Ford. This article discusses the potential impact of advanced language models like BERT on the future of work, including the possibility of automating certain tasks and creating new opportunities for human-AI collaboration.\n8. **"The Ethics of Language Models"** by Timothy B. Lee. This article discusses some of the ethical considerations surrounding the development and deployment|Here are some publications related to neural networks and large language models:\n\n1. "**Attention is All You Need" by Ashish Vaswani et al. (2017)** - This paper introduced the Transformer architecture, which relies solely on self-attention mechanisms to process input sequences.\n2. "**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**" by Jacob Devlin et al. (2019) - This paper presented a pre-trained language model called BERT, which achieved state-of-the-art results on a wide range of NLP tasks.\n3. "**Language Models for Natural Language Processing**" by Thomas N. Kwiatkowski et al. (2019) - This paper provided a comprehensive overview of language models and their applications in NLP.\n4. "**A Survey on Neural Machine Translation**" by Mohammad S. Al-Khateeb and Ali M. Al-Hamadi (2020) - This paper surveyed recent advances in neural machine translation, including the use of attention mechanisms and multi-task learning.\n5. "**Deep Learning for Natural Language Processing**" by Yoav Goldberg (2017) - This book provides a comprehensive introduction to deep learning techniques for NLP, covering topics such as word embeddings, recurrent neural networks, and sequence-to-sequence models.|
|return 3 publications about Computer Vision|and Robotics.\n\nHere are **three** relevant publications about computer vision and robotics:\n\n1. **"Deep Learning for Computer Vision and Robotics"** by Y. LeCun et al. (2016) - This paper provides an overview of the use of deep learning techniques in computer vision and robotics, including applications such as object recognition, image segmentation, and visual servoing.\n2. **"Visual Servoing for Robot Manipulation"** by S. H. Kim et al. (2018) - This paper discusses the use of computer vision and machine learning techniques for visual servoing, which is the process of controlling a robot\'s movements based on visual feedback from a camera. The authors present a survey of recent advances in visual servoing research and highlight some of the challenges and open research directions in this field.\n3. **"Object Detection and Recognition for Autonomous Robots"** by J. J. Rodriguez et al. (2019) - This paper reviews recent advances in object detection and recognition techniques for autonomous robots, with a focus on computer vision and machine learning methods. The authors discuss the use of deep learning techniques such as convolutional neural networks (CNNs) for object detection and recognition tasks, and highlight some of the challenges and limitations of these approaches.|Here are **three** publications related to Computer Vision based on the given context:\n\n1. **"Digital Gonioscopy Based on Three-dimensional Anterior-Segment OCT: An International Multicenter Study"** - This publication discusses the use of computer vision techniques for diagnosing and treating glaucoma.\n2. **"Current Status and research hotspots in the field of full endoscopic spine surgery: A bibliometric analysis"** - This publication examines the current state of computer vision research in the field of full endoscopic spine surgery.\n3. **"Effects of digital devices and online learning on computer vision syndrome in students during the COVID-19 era: an online questionnaire study"** - This publication investigates the impact of digital devices and online learning on computer vision syndrome in students.|
|Publications about Deep Learning for Stock Prediction|Deep learning has been increasingly applied to stock price prediction in recent years. Here are some research papers and articles that discuss the use of deep learning for stock prediction:\n\n1. **"Deep Learning for Stock Price Prediction"** by Y. C. Liu, et al. (2017) - This paper proposes a deep learning model based on a long short-term memory (LSTM) network to predict stock prices. The authors evaluate the performance of their model using a dataset of stock prices from the Taiwan Stock Exchange.\n2. **"Stock Price Prediction Using Deep Learning"** by S. R. Chakraborty, et al. (2018) - This paper presents a deep learning model based on a convolutional neural network (CNN) to predict stock prices. The authors use a dataset of stock prices from the Indian stock market and compare the performance of their model with traditional time series models.\n3. **"A Comparative Study of Deep Learning Techniques for Stock Price Prediction"** by A. K. Singh, et al. (2019) - This paper compares the performance of different deep learning models, including LSTM, CNN, and recurrent neural networks (RNNs), for stock price prediction. The authors use a dataset of stock prices from the US stock market and find that the LSTM model outperforms the other models.\n4. **"Deep Learning for Stock Market Prediction: A Survey"** by J. Zhang, et al. (2020) - This paper provides a comprehensive survey of deep learning techniques for stock market prediction. The authors discuss the advantages and limitations of different deep learning models and highlight the challenges and future directions in this area.\n5. **"Stock Price Prediction with Deep Learning"** by T. N. Khiat, et al. (2019) - This paper presents a deep learning model based on a multi-layer perceptron (MLP) network to predict stock prices. The authors use a dataset of stock prices from the US stock market and show that their model outperforms traditional time series models.\n\nThese papers and articles demonstrate the potential of deep learning for stock price prediction and provide insights into the challenges and opportunities in this area.'|Here are some publications related to deep learning for stock prediction:\n\n1. **Explainable natural language processing with matrix product states (2019)** - This paper proposes a deep learning model for stock price prediction that incorporates natural language processing techniques to extract relevant information from financial news articles.\n2. **The effect of short selling on volatility and jumps (2018)** - This paper examines the impact of short selling on stock price volatility and the occurrence of sudden price jumps in the US equity market.\n3. **Novelty Detection of a Rolling Bearing using Long Short-Term Memory Autoencoder (2019)** - This paper presents a deep learning approach for detecting novel patterns in rolling bearing data, which could be applied to stock price prediction.\n4. **Sugarcane Classification for On-Site Assessment Using Computer Vision (2019)** - This paper describes a computer vision system for classifying sugarcane based on its physical characteristics, which could potentially be used for predicting sugarcane prices and their impact on the stock market.\n\nI hope this helps! Let me know if you have any other questions.'|


## Result Analysis

- For the first prompt: The response from Llama13b hallucinates publication names and authors for papers 3, 4, and 5, while the response from the RAG pipeline hallucinates publication names and authors for papers 3 and 5, and authors for paper 4
- For the second prompt: The response from Llama13b hallucinates all publications and authors, whereas the response from RAG pipeline provides real publication names. However, the third publication is not related to Computer Vision
- For the third prompt: The response from Llama13b hallucinates all publications and authors, whereas the response from the RAG pipeline provides real publication names for the first two papers but hallucinates the third publication and the publication years for all publications|

From the above, it can be concluded that RAG can address some hallucination problems, such as publication names in this case, but there is still room for improvement regarding author names and publication release dates, which are still hallucinated by the model

## Conclusion
From the above , it can be concluded that RAG can address some hallucination problems, such as publication names in this case, but there is still room for improvement regarding author names and publication release dates, which are still hallucinated by the model

Ways to improve this study
- Increase the vector database knowledge
- Find additional techniques to enhance the RAG technique
- Clean the data to create a more robust vector database

## References
- Tutorial: [James Briggs's Better Llama 2 with Retrieval Augmented Generation (RAG)](https://www.youtube.com/watch?v=ypzmPwLH_Q4)
