## nlp-tutorial

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial` is a tutorial for who is studying NLP(Natural Language Processing) using **Pytorch**. Most of the models in NLP were implemented with less than **100 lines** of code.(except comments or blank lines)

## Dependencies

- Python 3.6+
- Pytorch 1.2.0+

## Curriculum - (Example Purpose)

#### 1. Basic Embedding Model

- 1-1. [NNLM(Neural Network Language Model)](https://github.com/graykode/nlp-tutorial/tree/master/1-1.NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Colab - [NNLM_Torch.ipynb](https://colab.research.google.com/drive/1-agQZoIOxaE68_SMaNGy35pz8ccWefps?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1AT4y1J7bv/
- 1-2. [Word2Vec(Skip-gram)](https://github.com/graykode/nlp-tutorial/tree/master/1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec_Torch(Softmax).ipynb](https://colab.research.google.com/drive/1rKNaAZwe3tdZMzKjOX6gP8nrQBhKxbFa?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV14z4y19777/
- 1-3. [FastText(Application Level)](https://github.com/graykode/nlp-tutorial/tree/master/1-3.FastText) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/drive/1vyLFapyCygGREa9jt11Zfy_DgTDGvGwm?usp=sharing)



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](https://github.com/graykode/nlp-tutorial/tree/master/2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - Colab -  [TextCNN_Torch.ipynb](https://colab.research.google.com/drive/13o8uID830WHL3rRZhXMoANc2XuqehRta?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1ip4y1U735/
- 2-2. DCNN(Dynamic Convolutional Neural Network)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](https://github.com/graykode/nlp-tutorial/tree/master/3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab -  [TextRNN_Torch.ipynb](https://colab.research.google.com/drive/1Krpcg9BNW97cXqmgnEcW2D05pDhLBMkA?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1iK4y147ff/
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab -  [TextLSTM_Torch.ipynb](https://colab.research.google.com/drive/1K75NsbkuejOzp2tfsXGDJxP-nQl9V0DC?usp=sharing)
- 3-3. [Bi-LSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**
  - Colab -  [Bi_LSTM_Torch.ipynb](https://colab.research.google.com/drive/1R_3_tk-AJ4kYzxv8xg3AO9rp7v6EO-1n?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1tf4y117hA/



#### 4. Attention Mechanism

- 4-1. [Seq2Seq](https://github.com/graykode/nlp-tutorial/tree/master/4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab -  [Seq2Seq_Torch.ipynb](https://colab.research.google.com/drive/18-pjFO8qYHOIqbb3aSReNpAbqZHCzLXq?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1Q5411W7zz/
- 4-2. [Seq2Seq with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab -  [Seq2Seq(Attention)_Torch.ipynb](https://colab.research.google.com/drive/1eObkehym2HauZo-NBYi39aAsWE1ujExk?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1op4y1U7ag/
- 4-3. [Bi-LSTM with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
  - Colab -  [Bi_LSTM(Attention)_Torch.ipynb](https://colab.research.google.com/drive/1RDXyIYPm6PWBWP4tVD85rkIo50clgyiQ?usp=sharing)



#### 5. Model based on Transformer

- 5-1.  [The Transformer](https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer_Torch.ipynb](https://colab.research.google.com/drive/15yTJSjZpYuIWzL9hSbyThHLer4iaJjBD?usp=sharing)
  - bilibili - https://www.bilibili.com/video/BV1mk4y1q7eK
- 5-2. [BERT](https://github.com/graykode/nlp-tutorial/tree/master/5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT_Torch.ipynb)

|           Model            |              Example               |
| :------------------------: | :--------------------------------: |
|            NNLM            |         Predict Next Word          |
|     Word2Vec(Softmax)      |   Embedding Words and Show Graph   |
|          TextCNN           |      Sentence Classification       |
|          TextRNN           |         Predict Next Step          |
|          TextLSTM          |            Autocomplete            |
|          Bi-LSTM           | Predict Next Word in Long Sentence |
|          Seq2Seq           |            Change Word             |
|   Seq2Seq with Attention   |             Translate              |
|   Bi-LSTM with Attention   |  Binary Sentiment Classification   |
|        Transformer         |             Translate              |
| Greedy Decoder Transformer |             Translate              |
|            BERT            |            how to train            |



## Author

- Tae Hwan Jung(Jeff Jung) @graykode，modify by [wmathor](https://github.com/wmathor)
- Author Email : nlkey2022@gmail.com
- Acknowledgements to [mojitok](http://mojitok.com/) as NLP Research Internship.
