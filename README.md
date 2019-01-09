## nlp-tutorial

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial` is a tutorial for who is studying NLP(Natural Language Processing) using **TensorFlow** and **Pytorch**. 

- Most of the models were implemented with less than 100 lines of code.
- You can also learn Tensorflow or Pytorch. 

## Curriculum

#### 1. Basic Embedding Model - (Example Purpose)

- 1-1. [NNLM(Neural Network Language Model)](https://github.com/graykode/nlp-tutorial/tree/master/1-1.%20NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 1-2. [Word2Vec(Skip-gram)](https://github.com/graykode/nlp-tutorial/tree/master/1-2.%20Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- 1-3. Glove - **Embedding Words and Show Graph**
- 1-4. FastText(Application Level) - **Sentence Classification**
  - Site : http://fasttext.cc
  - Paper : [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Usage : Google Colab



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](https://github.com/graykode/nlp-tutorial/tree/master/2-1.%20TextCNN) - **POS/NEG Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
- 2-2. DCNN(Dynamic Convolutional Neural Network)
- 2-3. DMCNN(Dynamic Multi-Pooling CNN)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](https://github.com/graykode/nlp-tutorial/tree/master/3-1.%20TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- 3-2. TextLSTM - **Autocomplete**
- 3-3. Bi-LSTM



#### 4. Attention Mechanism

- 4-1. [Sequence2Sequence](https://github.com/graykode/nlp-tutorial/tree/master/4-1.%20Seq2Seq) - **Translate**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
- 4-2. Attention Mechanism
- 4-3. The Transformer
- 4-4. Bi-LSTM with Attention



#### 5. Recursive Neural Network

- 5-1. Recursive Neural Network
- 5-2. MV-RNN(Matrix-Vector Recursive Neural Network)
- 5-3. RNTN(Recursive Neural Tensor Network)



#### 6. New Trend(Pre-Trained Model)

- 6-1. BERT



#### 7. ETC

- 7-1. TextRank
- 7-2. LDA(Latent Dirichlet Analysis)



## Author

- Tae Hwan Jung(Jeff Jung) @graykode
- Email : nlkey2022@gmail.com



## Reference Code

- [golbin/TensorFlow-Tutorials](https://github.com/golbin/TensorFlow-Tutorials)