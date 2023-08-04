THE CAPABILITIES OF LATENT DIRICHLET ALLOCATION - LDA (not traditional linear discriminant analysis)  
so what is the difference between topic analysis and topic modeling?  
Analysis - means clustering methods, used when you don't have labels and want to find/discover new topics.  
Modeling - like LDA, you don't have labels and you want to find new topics and learn about each text composition/distribution.  

For clustering methods, the document is assigned to only one topic, either A or B cluster if there are only two topics. what if the document belongs to both clusters?
we take the text composition of the document and assign how much percentage of topic A and topic B is related to the document.  

Working: 
* it will start with some topic assignments to the text,
* learn about the word composition of each topic.
* reiterate until it reaches the most reliable solution.
  * update step: topic-to-text assignment(topic distribution in the collection of documents), word-to-topic assignment(word distribution for each topic).


1. Imagine a word cloud that has a certain topic assigned to the cloud and words in the cloud are differentiated by their font size which represents the relative 
weight or contribution of each word to each topic.
2. LDA treats each document as a mixture of topics.
3. The LDA's default assumption is that the words in the document are not randomly aligned, but they are related. that is the the words are generated and put together that resembles the abstract topics.
4. Our goal is to reverse engineer this document generation process to detect which topics are responsible for the observed words.

* Short description:  
  * RANDOM ALLOCATION: randomly assign words to topics in a doc.  
  * INITIAL ESTIMATION: estimate topic and word distributions based on step1.  
  * REALLOCATION: evaluate using dist from step2 and reallocate words to respective topics.
  * RE-ESTIMATION: re-estimate dist based on reallocations from step3.
  * ITERATION: iterate step3 and 4 until the new estimation is the same as prev estimation or, until a certain number of iterations. 

Topic prob dist P( topic t / document d ) = (No 0f words in d allocated to t) / (total no of words in d)  
Word prob dist P( word w / topic t ) = (No of times w allocated to t in all d's) / (total no of occurrences of w)  

Update step: let's assume we have two topics, for a random given word "w"  
* Pupdate(w from t1) = Pprev(t1 / d1) * P(w/t1)  
* Pupdate(w from t2) = Prev(t2 / d1) * P(w/t2)  

If Pupdate(w from t1) > Pupdate(w from t2) assign the word to t1, else otherwise.  
After this step some of the allocations are changed, now you need to reestimate the topic prob. After multiple passes through the whole documents, this will reach to a stable distribution.  

```python
# the LDA model is available in gensim
import gensim
lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus, # [[(0,1), (1,3), (2,6), .....], [....]] array of tuples, where tuple (index, occurrences)
                                            id2word = id2word, # each word is mapped to unique id, a dict 
                                            num_topics = 2, # no of topics to allocate 
                                            random_state = 100,
                                            update_every = 1, # how often it should update topic distribution. 
                                            chunksize = 1000,
                                            passes = 10)

```

