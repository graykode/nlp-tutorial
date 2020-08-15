## nlp-tutorial

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial` is a tutorial for who is studying NLP(Natural Language Processing) using **Pytorch**. Most of the models in NLP were implemented with less than **100 lines** of code.(except comments or blank lines)

- [08-14-2020] Old TensorFlow v1 code is archived in [the archive folder](archive). For beginner readability, only pytorch version 1.0 or higher is supported.


## Curriculum - (Example Purpose)

#### 1. Basic Embedding Model

- 1-1. [NNLM(Neural Network Language Model)](1-1.NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Colab - [NNLM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM.ipynb)
- 1-2. [Word2Vec(Skip-gram)](1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram(Softmax).ipynb)
- 1-3. [FastText(Application Level)](1-3.FastText) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-3.FastText/FastText.ipynb)



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - [TextCNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN.ipynb)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN.ipynb)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM.ipynb)
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**
  - Colab - [Bi_LSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM.ipynb)



#### 4. Attention Mechanism

- 4-1. [Seq2Seq](4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq.ipynb)
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).ipynb)
- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
  - Colab - [Bi_LSTM(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention).ipynb)



#### 5. Model based on Transformer

- 5-1.  [The Transformer](5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb), [Transformer(Greedy_decoder).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder).ipynb)
- 5-2. [BERT](5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb)



## Dependencies

- Python 3.5+
- Pytorch 1.0.0+



## Author

- Tae Hwan Jung(Jeff Jung) @graykode
- Author Email : nlkey2022@gmail.com
- Acknowledgements to [mojitok](http://mojitok.com/) as NLP Research Internship.
