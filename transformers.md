we know that text is sequential data. Networks like LSTM contain feedback loops that allow information to propagate from one step to another. These types of architectures are mainly used for speech processing and time series. You can find more details in this amazing blog post by Andrej Karpathy: [The Unreasonable Effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).  

One of the main areas where RNN showed prominence is machine translation: one language to another. This type of machine translation model is built by using encoder-decoder architecture or seq-to-seq architectures(input and output are diff lengths).  
* The encoder encodes the information into a numerical sequence which is generally called a hidden state, the output of encoder.  
* This state is passed to the decoder. and the decoder outputs the sequence one at a time.
* The main drawback of this kind of architecture: 
  1) The hidden state creates an information bottleneck, which means it has to represent the entire meaning in a single hidden state. 
  2) If there is a long sequence, the information may be lost in encoding the whole sequence into a compressed format.
  3) This could be overcome by providing all of the encoder state's information to the decoder. This is called "ATTENTION". 

ATTENTION: 
* The main idea is instead of producing a single hidden state, the encoder(BERT) makes a representation of all the hidden states at each step that the decoder(GPT) can access. 
* As you thought of it by now, representing all the hidden states makes it a large matrix for the decoder input, you need a mechanism to prioritize which states to expose to the decoder.  
* This is where attention is prominent, the decoder assigns weight to the states of the encoder at every decoding step. 
* Another shortcoming  is, since the mechanism is sequential, we can not process this in parallel, The "SELF-ATTENTION" mechanism paved the way for parallel executions.  
* The basic idea is to operate attention on all states in the same layer of the neural network, and then fed into the feed-forward neural network. that is both encoder and decoder have self-attention mechanisms. this way it can be trained much faster than RNN models.
* As we do not have most of the resources to collect large amounts of datasets to create a model for machine translation, hugging face provides pre-trained models. that the weights of the entire architecture are provided, such that we can use the pre-trained version of the model on our available comparatively smaller dataset to fine-tune. this process is called "TRANSFER LEARNING".

GPT: As the name says it is a pre-trained transformer for generating text.  
BERT: This is also called masked-language-modeling. The objective is to find the randomly masked words in a text.  

HUGGING FACE: 

* pipelines: This pipeline() method from hugging face abstracts away all the steps need to convert raw text into set of predictions from a pre-trained/fine-tuned model.
* ```python
  import pandas as pd
  from transformers import pipeline
  classifier = pipeline('text-classification')
  text = """ I really love that movie"""
  pd.DataFrame(classifier(text))

  output: 	label	    score
          0	POSITIVE	0.999877
  ```
* In this the model is very confident that this is a positive sentiment. This demonstrates one of the common tasks to perform using hugging face pre-trained transformers. similarly, you can perform tasks such as: Named Entity Recognition, summarization, Question Answering, Translation, and text Generation.
* ```python
  # NER
  ner_tagger = pipeline("ner", aggregation_strategy="simple") 
  # Q-A
  reader = pipeline("question-answering")
  outputs = reader(question=question, context=text)
  # Translation english to german
  translator = pipeline("translation_en_to_de",
  model="Helsinki-NLP/opus-mt-en-de")
  outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
  # text generation
  generator = pipeline("text-generation")
  outputs = generator(prompt, max_length=200)
  ```

