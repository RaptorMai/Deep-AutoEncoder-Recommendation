# Deep-AutoEncoder-Recommendation

Autoencoder has been widely adopted into Collaborative Filtering (CF) for recommendation system. A classic CF problem is inferring the missing rating in an MxN matrix R where R(i, j) is the ratings given by the i<sup>th</sup> user to the j<sup>th</sup> item. This project is a Keras implementation of  AutoRec [1] and Deep AutoRec [2] with additional experiments such as the impact of default rating of users or ratings. 

The Dataset I used for this project is MovieLens 1M Dataset and can be downloaded from [here](<https://grouplens.org/datasets/movielens/1m/>). 

The preprocessing of the dataset can be found in this [Jupyter Notebook](<https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation/blob/master/Data_Preprocessing.ipynb>)

The implementation of models in Keras can be found in this [Jupyter Notebook](<https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation/blob/master/DeepAE_Rec.ipynb>)

## Reference

[1] Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." *Proceedings of the 24th International Conference on World Wide Web*. ACM, 2015

[2] Kuchaiev, Oleksii, and Boris Ginsburg. "Training deep autoencoders for collaborative filtering." *arXiv preprint arXiv:1708.01715* (2017).

[3]Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." *Proceedings of the Ninth ACM International Conference on Web Search and Data Mining*. ACM, 2016.

[4]Strub, Florian, Jérémie Mary, and Romaric Gaudel. "Hybrid collaborative filtering with autoencoders." *arXiv preprint arXiv:1603.00806* (2016).



## Github Reference

https://github.com/NVIDIA/DeepRecommender

<https://github.com/gtshs2/Autorec>

<https://github.com/henry0312/CDAE>

<https://github.com/cheungdaven/DeepRec>
