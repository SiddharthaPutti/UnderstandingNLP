# 60DaysofNLP
This repository will be my record of the NLP journey.

All this is going to be in lame/understandable language. 
Day 1: 

I know that, once the data is all preprocessed, the representation needs to be in numbers, not text anymore. so the text/tokens we have needs to be vectorized(vector/numbers format). 

There are many ways of representing text/tokens to vectors. some of them are: 
1) One-Hot Encoding.
2) Bag of Words.
3) N-Grams.
4) TF-IDF.  
    Distributed Representations:  
5) Word Embeddings.  

One-Hot Encoding:  
  * In One Hot encoding each word in the corpus is given a unique number/index. we simply put 1, 0 if the word is present in each document.  
  * which makes it a large sparse representation.  
  * The size of the vector is directly proportional to the size of the vocab.  
  * There is no relationship captured between the words.  
  
Bag Of Words:  
  * Dont get confused by the name(BoW). This method is generally used in text classification problems although it has its cons. let's see how it works in the first place.  
  * Instead of 0, 1 as One-Hot Encoding does, replace it with the number of times the word is seen in the particular document.  
  * now imagine that there are two documents, [2,4,0,3,2,1] and [2,3,1,4,4,1]. calculate the distance between both documents, which will be a lower value. that represents both documents might be similar.  
  * ```python
     from sklearn.feature_extraction.text import CountVectorizer  
     ```  
  
  * same as before, we have a large sparse vector, as the vocab increases.  
  * It doesn't capture similarity/relationship b/n different words. "I run", "I ran", and "I ate" All three vectors will be equally apart.  
  * No capturing of OOV. (out of vocab)  
  * No sequence info- word order information is lost.  
  
N-Grams:  
  * All of the above methods have words as independent units, that is no word order. To capture some context, we basically take N words at a time to calculate the count.  
  * if N=2, "The dog bit me", this document is going to be all the combination of words with length 2.  
  * By increasing the value of N, we can incorporate a larger context, however, it further increases the sparsity of the vector.  
  
  * It captures some context and word-order information, unlike BoW.  
  * Thus resulting vector captures some semantic similarity/relationship.  
  * This still doesn't address the OOV problem.  
  
TF-IDF:  
  * All the above methods treat all the words equally, let's give some bias to the most important words/token from the documents.  
  * Let's aim to give some importance to a particular word with respect to other words in the document, and also in the corpus.  
  * If a word appears in a document very often than in the other documents, then the word must be very important to that document.  
  * Also the importance must be increased in proportion to its frequency. given that its importance must be decreased in proportion to the word frequencies in other documents.  
  * Term-Freq: how often a word is seen in a document.  
    * tf = (no of occurrence of the word in doc) / (total no of terms in doc)  
  * Inv-Doc Freq: measures the importance of words across the corpus.  
    * IDF = (total no of docs) / (no of docs with that word in them)  
  * Now tf*idf  
  
  * ```python
     from sklearn.feature_extraction.text import TfidfVectorizer
     ```  
  
  * used in text classification and information retrieval.  
  * We can use tf-idf to calculate similarity b/n docs using Euclidean/cosine.  
  
  * same as previous methods, large sparse vectors.  
  * Still can not address OOV.  
  
Distributed Representations:  
  * Word Embeddings:  
    These are NN-based methods for dense(low dimensional) representations of words.  
    given the word USA, it must be similar to Canada, China, etc... this is usually called distributional similarity. these can be considered distributionally similar.  
    One of the popular models to capture distributional similarity is "Word2Vec".  
        * This works well to capture analogies like King- Man+Woman~ Queen.  
        * This representation is also called Embedding. This ensures the vector is dense(very less sparse- less 0's)  
        * if two different words occur in a similar context, then it's highly likely that the meanings of the words are also similar.  
        *  Some of the other pre-trained Embeddings are GloVe by stanford, fasttext by facebook...  
        * fortunately, embeddings like word2vec have already pre-trained versions. you can load the pre-trained versions of word2vec and get the embeddings of words in your required length.  

    You can also train your own word embeddings, for this there are two main proposed architectures:  
        * CBOW - continuous bag of words  
            * This is like filling the blanks kind of learning: the model tries to predict the center word given the rest of the context. In this method, the model tries to assign probabilities in such a way that the context makes sense.  
            * It takes every word in the corpus as the target and assigns prob to every word with respect to a given context.
            * Input:
                * take a fixed window size for example 10, that is 20 words at a time from a document. 10 before and 10 after the center word  
                * create a one-hot representation of each word, the matrix for a window size of 10 will now be: (window_size *2, vocab_size)  
                * slide the window for the next 20 words, that is slide for one word at a time, this will again create a matrix of size: (window_size *2, vocab_size)  
                * continue this process until the end of the document, pad the document if necessary to include all the context. this will create a number of matrices, concatenate them all to make it a large sparse matrix, and serve as input to NN.
                * if you are using embeddings like word2vec, the process is similar, but the size of the matrix changes to (window_size*2, embedding_dim), that is each word will have a fixed embedding length.  
        * SkipGram: given the center word predict the context words.  
    
    
  
  
  
