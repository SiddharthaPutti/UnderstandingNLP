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
  * update step: topic-to-text assignment, word-to-topic assignment.


