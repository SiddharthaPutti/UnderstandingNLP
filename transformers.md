we know that text is sequential data. Networks like LSTM contain feedback loops that allow information to propagate from one step to another. These types of architectures are mainly used for speech processing and time series. You can find more details in this amazing blog post by Andrej Karpathy: [The Unreasonable Effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).  

One of the main areas where RNN showed prominence is machine translation: one language to another. This type of machine translation model is built by using encoder-decoder architecture or seq-to-seq architectures(input and output are diff lengths).  
  The encoder encodes the information into a numerical sequence which is generally called a hidden state, the output of encoder.  
  This state is passed to the decoder. and the decoder outputs the sequence one at a time.
  The main drawback of this kind of architecture: 
    The hidden state creates an information bottleneck, which means it has to represent the entire meaning in a single hidden state. 
    If there is a long sequence, the information may be lost in encoding the whole sequence into a compressed format.
    This could be overcome by providing all of the encoder state's information to the decoder. This is called "ATTENTION". 

ATTENTION: 
  The main idea is instead of producing a single hidden state, the encoder makes a representation of all the hidden states at each step that the decoder can access. 
  As you thought of it by now, representing all the hidden states makes it a large matrix for the decoder input, you need a mechanism to prioritize which states to expose to the decoder.  
  This is where attention is prominent, the decoder assigns weight to the states of the encoder at every decoding step. 
  Another shortcoming  is, since the mechanism is sequential, we can process this in parallel, The self-attention mechanism paved the way for parallel executions.  
    the basic idea is to operate attention on all states in the same layer of the neural network, and then fed into the feed-forward neural network. that is both encoder and decoder have self-attention mechanisms. this way it can be trained much faster than RNN models. 
