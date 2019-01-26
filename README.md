## nlp-tutorial

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial` is a tutorial for who is studying NLP(Natural Language Processing) using **TensorFlow** and **Pytorch**. 

- Most of the models in NLP were implemented with less than **100 lines** of code.(except comments or blank lines)
- You can also learn Tensorflow or Pytorch. 

## Curriculum

#### 1. Basic Embedding Model - (Example Purpose)

- 1-1. [NNLM(Neural Network Language Model)](https://github.com/graykode/nlp-tutorial/tree/master/1-1.NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 1-2. [Word2Vec(Skip-gram)](https://github.com/graykode/nlp-tutorial/tree/master/1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- 1-3. FastText(Application Level) - **Sentence Classification**
  - Site : http://fasttext.cc
  - Paper : [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Usage : Google Colab



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](https://github.com/graykode/nlp-tutorial/tree/master/2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
- 2-2. DCNN(Dynamic Convolutional Neural Network)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](https://github.com/graykode/nlp-tutorial/tree/master/3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 3-3. Bi-LSTM - **Predict Long Next Step Word**



#### 4. Attention Mechanism

- 4-1. [Sequence2Sequence](https://github.com/graykode/nlp-tutorial/tree/master/4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
- 4-2. [Seq2Seq with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
- 4-3. [Bi-LSTM with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
- 4-4. [The Transformer](https://github.com/graykode/nlp-tutorial/tree/master/4-4.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)



#### 5. Model based on Transformer

- 6-1. BERT



## Author

- Tae Hwan Jung(Jeff Jung) @graykode
- Email : nlkey2022@gmail.com



## Reference Code

- [golbin/TensorFlow-Tutorials](https://github.com/golbin/TensorFlow-Tutorials)