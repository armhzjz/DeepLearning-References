# DeepLearning References
This is just a place to save the deep learning references that I believe are valueable and helpful
<br>
<br>

## Neral Networks - the basics
<br>

### Papers
| Paper 	| Authors 	| Application 	| comment 	|
|:-:	|:-:	|:-:	|-	|
| [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) 	| Yann LeCun 	|  	| :point_left: 	|
| [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) 	| Yoshua Bengio 	| - 	| :point_left:	|
| [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) 	| YSergey Ioffe, Christian Szegedy 	| - 	| :point_left:	|
| [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 	| Xavier Glorot, Yoshua Bengio 	| - 	| :point_left:	|
| [Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) 	| Laurens van der Maaten, Geoffrey Hinton 	| - 	| :point_left:	|
| [Accelerating t-SNE using Tree-Based Algorithms](http://jmlr.org/papers/v15/vandermaaten14a.html) 	| Laurens van der Maaten 	| - 	| :point_left:	|

### Articles and other resources
* [Tradeoff batch size vs. number of iterations to train a neural network](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)
* [Deep Learning Book](http://www.deeplearningbook.org/) :point_left:
* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) :point_left:
* [Tuning the learning rate in Gradient Descent](http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/) :point_left:
* [Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/) :point_left:
* [Visualizing MNIST: An Exploration of Dimensionality Reduction](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) :point_left:

<br>
<br>

## CNNs - Image & Object detection
<br>

### Papers
| Paper 	| Authors 	| Application 	| comment 	|
|:-:	|:-:	|:-:	|-	|
| [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 	| Leon A. Gatys, Alexander S. Ecker, Matthias Bethge 	| Style Transfer 	|  	|
| [Depth Map Prediction from a Single Imageusing a Multi-Scale Deep Network](https://cs.nyu.edu/~deigen/depth/depth_nips14.pdf) 	| David Eigen, Christian Puhrsch, Rob Fergus 	| - 	|  	|
| [Dynamic Routing Between Capsules](https://video.udacity-data.com/topher/2018/November/5bfdca4f_dynamic-routing/dynamic-routing.pdf) 	| Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton 	| - 	|  	|
| [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) 	| Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger 	| - 	| My own implementation of Densenet as a python module can be found [here](https://github.com/armhzjz/DenseNet).	|
| [Gradient Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) 	| Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner 	| - 	| 	|
| [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) 	| Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson 	| - 	| :point_left:	|
| [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) 	| Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 	| - 	| :point_left:	|
| [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf) 	| Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, Demetri Terzopoulos 	| - 	| 	|
| [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) 	| Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 	| - 	| :point_left:	|
| [The Importance of Skip Connections in Biomedical Image Segmentation](https://arxiv.org/pdf/1608.04117.pdf) 	| Michal Drozdzal, Eugene Vorontsov, Gabriel Chartrand, Samuel Kadoury, Chris Pal 	| - 	| :point_left:	|


<br>

### Articles and other resources
* [Review of Deep Learning Algorithms for Object Detection](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
* [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://cs.nyu.edu/~deigen/depth/)
* [Capsule Networks](https://cezannec.github.io/Capsule_Networks/)
* [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
* [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/#conv)
* [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035) :point_left:

<br>
<br>

## Recurrent Neural Networks
<br>

### Papers

| Paper 	| Authors 	| Application 	| comment 	|
|:-:	|:-:	|:-:	|-	|
| [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) 	| Andrej Karpathy, Justin Johnson, Li Fei-Fei 	| - 	| :point_left: 	|
| [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) 	| Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio 	| - 	|  	|
| [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf) 	| Rafal Jozefowicz,  Wojciech Zaremba, Ilya Sutskever 	| - 	|  	|
| [LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf) 	| Klaus Greff, Rupesh K. Srivastava, Jan Koutn ́ık, Bas R. Steunebrink, J ̈urgen Schmidhuber  	| - 	|  	|
| [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf) 	| Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever 	| - 	|  	|
| [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906v2) 	| Denny Britz, Anna Goldie, Minh-Thang Luong, Quoc Le 	| - 	|  	|
| [WAVENET: A GENERATIVEMODEL FORRAWAUDIO](https://arxiv.org/pdf/1609.03499.pdf) 	| Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu 	| Deep generative model of raw audio waveforms 	|  	|
| [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) 	| Siwei Lai, Kang Liu, Liheng Xu, Jun Zhao 	| -	|  	|
| [Systematic evaluation of CNN advances on the ImageNet by](https://arxiv.org/abs/1606.02228) 	| Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas 	| -	| :point_left: 	|
| [Efficient Estimation of Word Representations inVector Space](https://arxiv.org/abs/1301.3781) 	| Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean 	| -	| :point_left: 	|
| [Distributed Representations of Words and Phrasesand their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 	| Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean 	| -	| :point_left: 	|
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) 	| Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio 	| -	|  	|
| [Learning Phrase Representations using RNN Encoder–Decoderfor Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) 	| Kyunghyun Cho, Bart van Merri ̈enboe, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio 	| -	|  	|
| [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) 	| Minh-Thang Luong, Hieu Pham, Christopher D. Manning 	| -	|  	|

<br>

### Example RNN Architectures

| Application 	| Cell 	| Layers 	| Size 	| Vocabulary 	|  	| Learning Rate 	| Paper 	|
|-	|-	|-	|-	|-	|-	|-	|-	|
| Speech Recognition (large vocabulary) 	| LSTM 	| 5, 7 	| 600, 1000 	| 82K, 500K 	| -- 	| -- 	| [Neural Speech Recognizer: Acoustic-to-Word LSTM Model for Large Vocabulary Speech Recognition](https://arxiv.org/abs/1610.09975) 	|
| Speech Recognition 	| LSTM 	| 1, 3, 5 	| 250 	| -- 	| -- 	| 0.001 	| [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778) 	|
| Machine Translation (seq2seq) 	| LSTM 	| 4 	| 1000 	| Source: 160K, Target: 80K 	| 1,000 	| -- 	| [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) 	|
| Image Captioning 	| LSTM 	| -- 	| 512 	| -- 	| 512 	| (fixed) 	| [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) 	|
| Image Generation 	| LSTM 	| -- 	| 256, 400, 800 	| -- 	| -- 	| -- 	| [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623) 	|
| Question Answering 	| LSTM 	| 2 	| 500 	| -- 	| 300 	| -- 	| [A Long Short-Term Memory Model for Answer Sentence Selection in Question Answering](https://www.aclweb.org/anthology/P15-2116/) 	|
| Text Summarization 	| GRU 	|  	| 200 	| Source:  119K, Target:  68K 	| 100 	| 0.001 	| [Sequence-to-Sequence RNNs for Text Summarization](https://www.semanticscholar.org/paper/Sequence-to-Sequence-RNNs-for-Text-Summarization-Nallapati-Xiang/221ef0a2f185036c06f9fb089109ded5c888c4c6?p2df) 	|

<br>

### Articles and other resources
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) :point_left:
* [Preprocessing text before use RNN](https://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn) :point_left:
* [Sentiment Analysis - A _very good_ tutorial!](https://github.com/jadianes/data-science-your-way/tree/master/04-sentiment-analysis) :point_left:
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [WaveNet: A generative model for raw audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
* [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) :point_left:
* [Applying word2vec to Recommenders and Advertising](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)
* [Natural Language Processing Key Terms, Explained](https://www.kdnuggets.com/2017/02/natural-language-processing-key-terms-explained.html)
* [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)

<br>
<br>

## Generative Adversarial Networks
<br>

### Papers

| Paper 	| Authors 	| Application 	| comment 	|
|:-:	|:-:	|:-:	|-	|
| [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) | Ian J. Goodfellow,  Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio | - 	| :point_left: 	|
| [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf) | Alec Radford, Luke Metz | - 	| :point_left: 	|
| [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) | Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen | - 	| :point_left: 	|
| [Fine-Grained Car Detection for Visual Census Estimation](https://arxiv.org/pdf/1709.02480.pdf) | Tim Gebru, Jonathan Krause, Yilun Wang, Duyun Chen, Jia Deng, Li Fei Fei | - 	| 	|
| [CycleGAN Face-off](https://arxiv.org/pdf/1712.03451.pdf) | Xiaohan Jin, Ye Qi Shangxuan Wu | - 	| :point_left:	|
| [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) | Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros | - 	| :point_left:	|
| [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf) | Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro | - 	| 	|
| [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf) | Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros | - 	| 	|
| [Augmented CycleGAN: Learning Many-to-Many Mappings from Unpaired Data](https://arxiv.org/abs/1802.10151) | Amjad Almahairi, Sai Rajeswar, Alessandro Sordoni, Philip Bachman, Aaron Courville | - 	| :point_left:	|
| [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) | Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo | - 	| :point_left:	|
| [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076) | Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley | - 	| :point_left:	|
| [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585) | Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro | - 	| 	|

<br>

### Articles and other resource
 * [Improved GAN (Semi-supervised GAN)](https://github.com/Sleepychord/ImprovedGAN-pytorch)
 * [Semi-Supervised Learning](https://www.youtube.com/watch?v=_LRpHPxZaX0&feature=emb_logo) :point_left:
 * [Attacking Machine Learning with Adversarial Examples](https://openai.com/blog/adversarial-example-research/)
 * [iGAN: Interactive Image Generation via Generative Adversarial Networks](https://github.com/junyanz/iGAN/blob/master/README.md)
 * [Image to Image Demo](https://affinelayer.com/pixsrv/)
 * [StarGAN - Official PyTorch Implementation](https://github.com/yunjey/StarGAN) :point_left:
 * [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)
 
<br>
<br>

## Deep Reinforcement Learning
<br>

### Articles and other resource
* [Deep Traffic](https://selfdrivingcars.mit.edu/deeptraffic/)
* [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
* [SnakeAI](https://github.com/greerviau/SnakeAI)
* [Markov Chain Monte Carlo Without all the Bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)

<br>
<br>

## Optimizers
<br>

### Papers

| Paper 	| Authors 	| Application 	| comment 	|
|:-:	|:-:	|:-:	|-	|
| [SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS](https://arxiv.org/pdf/1608.03983.pdf) | Ilya Loshchilov & Frank Hutter | - 	| :point_left: 	|

<br>

### Articles and other resource
* [Optimizers Explained - Adam, Momentum and Stochastic Gradient Descent](https://mlfromscratch.com/optimizers-explained/#/) :point_left:
* [Tuning the learning rate in Gradient Descent](http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/) :point_left:

<br>
<br>

## Miscellaneous
<br>

* [Loss functions](https://lossfunctions.tumblr.com/)
* [deepmind research papers](https://deepmind.com/research?filters=%7B%22tags%22:%5B%22Speech%22%5D%7D)
* [Reading Barcodes on Hooves: How Deep Learning Is Helping Save Endangered Zebras](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)
* [Tesla autopilot](https://www.tesla.com/autopilotAI)
* [Attacking Machine Learning with Adversarial Examples](https://openai.com/blog/adversarial-example-research/)
* [AI, Deep Learning, and Machine Learning: A Primer](https://www.youtube.com/watch?v=ht6fLrar91U&feature=youtu.be)
* [Deep Learning State of the Art (2020) | MIT Deep Learning Series](https://www.youtube.com/watch?v=0VH1Lim8gL8) :point_left:
* [Better Deep Learning - Train Faster, Reduce Overfitting, and Make Better Predictions](https://machinelearningmastery.com/better-deep-learning/#packages)
* [How to attack a machine learning model?](https://www.kaggle.com/allunia/how-to-attack-a-machine-learning-model)
* [Reading game frames in Python with OpenCV - Python Plays GTA V](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)
