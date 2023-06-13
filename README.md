# 60DaysofNLP
This repository will be my record of NLP journey.

All this is going to be in lame/understandable language. 
Day 1: 

I know that, once the data is all preprocessed, the representation needs to be in numbers, not text anymnore. so the text/tokens we have needs to be vectorized(vector/numbers format). 

There are many ways of representing text/tokens to vector. some of them are: 
1) One-Hot Encoding.
2) Bag of Words.
3) N-Grams.
4) TF-IDF.  
    Distributed Representations:  
5) Word Embeddings.  

One-Hot Encoding:  
  * In One Hot encoding each word in the corpus is given a unique number/index. we simply put 1, 0 if the word is present in each document.  
  * which makes it a large sparse representation.  
  * The size of the vector directly proportional to size of the vocab.  
  * There is no relationship captured between the words.  
  
Bag Of Words:  
  * Dont get confused by the name(BoW). This method is generally used in text classification problem although it have its cons. lets see how it works in the first place.  
  * Instead of 0, 1 as One-Hot Encoding does, replace it with number of times the word seen in the perticular document.  
  * now imagine that there are two documents, [2,4,0,3,2,1] and [2,3,1,4,4,1]. calculate the distance between both documents, which will be lower vlaue. that represents both documents might be similar.  
  * """ from sklearn.feature_extraction.text import CountVectorizer """  
  
  * same as before, we have a large sparse vectors, as the vocab increases.  
  * It doesnt capture similarity/relationship b/n different words. "I run", "I ran", "I ate" all three vectors will be equally apart.  
  * No capturing of OOV. (out of vocab)  
  * No sequence info- word order information is lost.  
  
N-Grams:  
  * All of the above methods have words as independent units, that is no word ordering. To capture some context, we basically take N words at a time to calculate the count.  
  * if N=2, "The dog bit me", this document is going to be all the combination of words with lenght 2.  
  * By increasing the vlaue of N, we can incoporate larger context, however it further increases the sparsity of the vector.  
  
  * It captures some context and word-order information unlike BoW.  
  * Thus resulting vector captures some sematic similarity/relationship.  
  * This still doesnt address OOV problem.  
  
TF-IDF:  
  * All the above methods treat all the words equally, lets give some bias to the most important words/token from the documents.  
  * Lets aim to give some importance to a perticular word with respect to other words in the document, and also in the corpus.  
  * If a word appears in a document very often than in the other documents, then the word must be very important to that document.  
  * Also the importance must be increased in proportion to its frequencey. given that its importance must be decreased in proportion to the word frequencies in other documents.  
  * Term-Freq: how often a word seen in a docuemnt.  
    * tf = (no of occurance of word in doc) / (total no of terms in doc)  
  * Inv-Doc Freq: measures the importance of word across the corpus.  
    * idf = (total no of docs) / (no of docs with that word in them)  
  * Now tf*idf  
  
  * """ from sklearn.feature_extraction.text import TfidfVectorizer """  
  
  * used in text classification and information retrieval.  
  * we can use tf-idf to calculate similarity b/n docs using euclidean/cosine.  
  
  * same as previous methods, large sparse vectors.  
  * Still can not address OOV.  
  
Distributed Representations:  
  * Word Embeddings:  
    These are NN based methods for dense(low dimensional) representations of words.  
    given the word USA, it must be similar to Canada, China etc... this usually called as distributional similarity. these can be considered as distributionally similar.  
    One of the popular model to capture distibutional similarity is "Word2Vec".  
        * This works well to capture anologies like king- Man+Woman~ Queen.
        * This representaion is also called as Embeddings. This ensures vector to be dense(very less sparse- less 0's)
        * if two different words occurs in similar context, then its highly likely that meaning of the words are aslo similar.  
        *  
  
  
  
