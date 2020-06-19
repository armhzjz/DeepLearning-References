# DeepLearning References
This is just a place to save the deep learning references that I believe are valueable and helpful
<br>
<br>

## CNNs - Image & Object detection
<br>

### Papers
| Paper 	| Authors 	| Application 	| Checked 	|
|:-:	|:-:	|:-:	|-	|
| [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 	| Leon A. Gatys, Alexander S. Ecker, Matthias Bethge 	| Style Transfer 	|  	|
| [Depth Map Prediction from a Single Imageusing a Multi-Scale Deep Network](https://cs.nyu.edu/~deigen/depth/depth_nips14.pdf) 	| David Eigen, Christian Puhrsch, Rob Fergus 	| - 	|  	|
| [Dynamic Routing Between Capsules](https://video.udacity-data.com/topher/2018/November/5bfdca4f_dynamic-routing/dynamic-routing.pdf) 	| Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton 	| - 	|  	|

<br>

### Articles and other resources
* [Deep Learning Book](http://www.deeplearningbook.org/)
* [Review of Deep Learning Algorithms for Object Detection](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)
* [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://cs.nyu.edu/~deigen/depth/)
* [Capsule Networks](https://cezannec.github.io/Capsule_Networks/)

<br>

## Recurrent Neural Networks
<br>

### Papers

| Paper 	| Authors 	| Application 	| Checked 	|
|:-:	|:-:	|:-:	|-	|
| Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling 	| Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio 	| - 	|  	|
| An Empirical Exploration of Recurrent Network Architectures 	| Rafal Jozefowicz,  Wojciech Zaremba, Ilya Sutskever 	| - 	|  	|
| Visualizing and Understanding Recurrent Networks 	| Andrej Karpathy, Justin Johnson, Li Fei-Fei 	| - 	|  	|
| LSTM: A Search Space Odyssey 	| Klaus Greff, Rupesh K. Srivastava, Jan Koutn ́ık, Bas R. Steunebrink, J ̈urgen Schmidhuber  	| - 	|  	|
| An Empirical Exploration of Recurrent Network Architectures 	| Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever 	| - 	|  	|
| Massive Exploration of Neural Machine Translation Architectures 	| Denny Britz, Anna Goldie, Minh-Thang Luong, Quoc Le 	| - 	|  	|
| [WAVENET: A GENERATIVEMODEL FORRAWAUDIO](https://arxiv.org/pdf/1609.03499.pdf) 	| Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu 	| Deep generative model of raw audio waveforms 	|  	|

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
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [WaveNet: A generative model for raw audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)

<br>
<br>

## Miscellaneous
<br>

* [Loss functions](https://lossfunctions.tumblr.com/)
* [Tuning the learning rate in Gradient Descent](http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/)
* [deepmind research papers](https://deepmind.com/research?filters=%7B%22tags%22:%5B%22Speech%22%5D%7D)
* [Reading Barcodes on Hooves: How Deep Learning Is Helping Save Endangered Zebras](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)
* [Tesla autopilot](https://www.tesla.com/autopilotAI)
