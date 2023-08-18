
This repository will be my record of the NLP journey.

All this is going to be in lame/understandable language. 
 

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
    * If two different words occur in a similar context, then it's highly likely that the meanings of the words are also similar.  
    *  Some of the other pre-trained Embeddings are GloVe by Stanford, fasttext by Facebook...  
    * fortunately, embeddings like word2vec have already pre-trained versions. you can load the pre-trained versions of word2vec and get the embeddings of words in your required length.  

    You can also train your own word embeddings, for this there are two main proposed architectures:  
    * CBOW - continuous bag of words  
        * This is like filling the blanks kind of learning: the model tries to predict the center word given the rest of the context. In this method, the model tries to assign probabilities in such a way that the context makes sense.  
        * It takes every word in the corpus as the target and assigns prob to every word with respect to a given context.
        * Input:
            * take a fixed window size for example 10, that is 20 words at a time from a document. 10 before and 10 after the center word  
            * Create a one-hot(or indexed) representation of each word, the matrix for a window size of 10 will now be: (window_size *2, vocab_size) {except the center word}  
            * slide the window for the next 20 words, that is slide for one word at a time, this will again create a matrix of size: (window_size *2, vocab_size)  
            * continue this process until the end of the document, pad the document if necessary to include all the context. this will create a number of matrices, concatenate them all to make it a large sparse matrix, and serve as input to NN.
            * if you are using embeddings like word2vec, the process is similar, but the size of the matrix changes to (window_size*2, embedding_dim), that is each word will have a fixed embedding length.
            * Output will be (the_no_of_concatenated_windows, vocab), for each window predict the prob distribution of the entire vocab and select the highest prob word.
    ```python
    # vocabulary 
    vocab = set(raw_text) 

    # converting words to indexes and indexes to words, just creating of dictionary to map them
    word_to_index = {word:index for index, word in enumerate(vocab)}
    index_to_word = {index:word for index, word in enumerate(vocab)}
    
    def create_context(context, word_to_index):
        ids = [word_to_index[w] for w in context]
        return ids

    # appending to data, creating inputs 
    data = []
    for i in range(window_size, len(raw_text) -window_size):
        context = raw_text[i-window_size: i] + raw_text[i+1: raw_text[i+window_size+1]
        target = raw_text[i]
        data.append((context,target))
              
    # for each epoch run the following in training loop 
    for context, target in data: 
        context_vector = create_context(context, word_to_index)
        log_pob = model(context_vector)
        loss = #
    
    ```
    now that you have trained a CBOW model, the word embeddings are the weights of the hidden layer. the first row of the hidden layer representation is the word embeddings for the 0th word. the hidden layer dims (vocab_size, embedding_dim)
    
    * SkipGram: given the center word predict the context words.  
        * we run a sliding window of size 2k+1, k words before the center word and k words after the center word, +1 for the center word.  
        * Similar to the CBOW, the word embeddings are weights of the hidden layer.
    ```python
    import numpy as np
    
    sentence = "The quick brown fox jumps over the lazy dog"
    window_size = 5
    words = sentence.lower().split()
    words = ['<pad>'] * window_size + words + ['<pad>']* window_size
    
    vocab = sorted(set(words))
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    X_train = [] 
    y_train = [] 
    
    for i in range(window_size, len(words)-window_size):
        target_word = words[i]
        target_idx = word2idx[target_word]
    
        context_words = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(words):
                context_word = words[j]
                context_idx = word2idx[context_word]
                context_words.append(context_idx)
    
        X_train.append(target_idx)
        y_train.append(context_words)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    for i in range(len(X_train)):
        target_word = idx2word[X_train[i]]
        context_word = [idx2word[word] for word in y_train[i]]
        print(f"Target word: {target_word}, Context word: {context_word}")
    
    ```

    ```
    # the output for this code is: 
    X_train: array([8, 7, 1, 3, 4, 6, 8, 5, 2])
    y_train: array([[0, 0, 0, 0, 0, 7, 1, 3, 4, 6],
                   [0, 0, 0, 0, 8, 1, 3, 4, 6, 8],
                   [0, 0, 0, 8, 7, 3, 4, 6, 8, 5],
                   [0, 0, 8, 7, 1, 4, 6, 8, 5, 2],
                   [0, 8, 7, 1, 3, 6, 8, 5, 2, 0],
                   [8, 7, 1, 3, 4, 8, 5, 2, 0, 0],
                   [7, 1, 3, 4, 6, 5, 2, 0, 0, 0],
                   [1, 3, 4, 6, 8, 2, 0, 0, 0, 0],
                   [3, 4, 6, 8, 5, 0, 0, 0, 0, 0]])


    Target word: the, Context word: ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'quick', 'brown', 'fox', 'jumps', 'over']
    Target word: quick, Context word: ['<pad>', '<pad>', '<pad>', '<pad>', 'the', 'brown', 'fox', 'jumps', 'over', 'the']
    Target word: brown, Context word: ['<pad>', '<pad>', '<pad>', 'the', 'quick', 'fox', 'jumps', 'over', 'the', 'lazy']
    Target word: fox, Context word: ['<pad>', '<pad>', 'the', 'quick', 'brown', 'jumps', 'over', 'the', 'lazy', 'dog']
    Target word: jumps, Context word: ['<pad>', 'the', 'quick', 'brown', 'fox', 'over', 'the', 'lazy', 'dog', '<pad>']
    Target word: over, Context word: ['the', 'quick', 'brown', 'fox', 'jumps', 'the', 'lazy', 'dog', '<pad>', '<pad>']
    Target word: the, Context word: ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', '<pad>', '<pad>', '<pad>']
    Target word: lazy, Context word: ['brown', 'fox', 'jumps', 'over', 'the', 'dog', '<pad>', '<pad>', '<pad>', '<pad>']
    Target word: dog, Context word: ['fox', 'jumps', 'over', 'the', 'lazy', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']

    ```
    Or some of the implementations include the x_train vector to have for every word in context, the x_train will have the target word repeated.  
    ```
    # soething like this: 
    Target word: the, Context word: quick
    Target word: the, Context word: brown
    Target word: the, Context word: fox
    Target word: the, Context word: jumps
    Target word: the, Context word: over
    Target word: quick, Context word: the
    Target word: quick, Context word: brown
    Target word: quick, Context word: fox
    Target word: quick, Context word: jumps
    Target word: quick, Context word: over
    Target word: quick, Context word: the
    Target word: brown, Context word: the
    Target word: brown, Context word: quick
    Target word: brown, Context word: fox
    Target word: brown, Context word: jumps
    Target word: brown, Context word: over
    Target word: brown, Context word: the
    Target word: brown, Context word: lazy
    ```  

    There are pre-trained word embeddings like Word2Vec and Doc2vec implemented in gensim.

    ```python
    from gensim.models import Word2Vec
    from gensim.test.utils import common_texts

    word2vec_model = Word2Vec(common_texts, size = 10, window =5, min_count =1, workers =4)
    print(word2vec_model.wv['like']) # This will print the vector representation of word "like" 
    print(word2vec_mode.wv.most_similar('computer', topn =6) # this will print top 6 most similar words to computer
    ```

    * One of the challenges every method discussed faces is OOV. One way is to remove all the OOV that are not in the corpus vocabulary or the fastText from Facebook takes care of words by their morphological representations, for example, the word "gregarious", is converted to n-grams representation of "gre", "ega"... etc,
    * we can use a pre-trained version of [fasttext](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html#sphx-glr-auto-examples-tutorials-run-fasttext-py) or train our data on fasttext using gensim. 
    
    
    
  
  
  
