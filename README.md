NLP practice

1. Folder of news <br>
   kaggle link: https://www.kaggle.com/datasets/kevinmorgado/spanish-news-classification. <br>
   In this Jupyter Notebook I practice the Tokenization, Stemming and Lemmatization tecniques, using the naive bayes model (MultinomialNB). Creating trainings with stemming and lemmatization tecniques so I can      compare both of them with the accuracy results, knowing the lemmatization tecnique have a higher computational cost, but in this case the results were similar, so depends of the porpuse we can select a different tecnique of tokenization. <br>
2. Folder of movies recomendations (recomendacion_peliculas) <br>
    kaggle link: https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset. <br>
   In this Jupyter Notebook I practice the TF-IDF method to visualice the Cosine way to calculate the similarity between vectors (cosine_similarity), using matplotlib library to see the type of graph, in this case    a logaritming graph, I can do this by taking a movie from this dataset and compare it with the rest of the movies and selecting the first ten movies for movies recommendations.


|  Image  | Description |
|---|---|
|  ![image](https://github.com/user-attachments/assets/502a2548-44f4-4976-9e03-a329f36d2dc8) |  This graph has the data out of order, showing the data in csv original order. the dataset has around 5000 columns. |
|  ![image](https://github.com/user-attachments/assets/09b505e3-dc3a-4a7d-868f-8c47f39a483d) | And this graph has a logaritming form due to ordering data from highest to lowest similarity of vectors from this movies   |

3. Folder of similarity <br>
   kaggle link: https://www.kaggle.com/datasets/rtatman/pretrained-word-vectors-for-spanish  <br>
4. Folder Word2Vec: <br>
   There are some texts for training <br>
5. Folder Text Clasificator <br>
   
6. Folder or text generator <br>
   In this practice i created a text generator, using a txt documents with poems of Benedetti, aplying second order Markov model, that means a way to  store more than 1 state, (its posible make mor than a 2 order  but the computational cost increase, for this practice only use 2 for aply all a learn.) more than a normal Markov model wich usually just use the actual state for make a predictions of the next word base on a probability to appear a next word,  this is using for give sense to a sentence like an generator text, so the matrix was a tridimensional space, and we need take 2 points in time, t_1 and t_2 (new states) for this case. Also for the probabilities i count the frecuency of ocurrence of some words together a another word, this help me to calculate the different probabilities for a different word sequences and to know (in the case of my small dataset) which sequences are most common and wich words had the highest probability to apper next to the before word, so my text generator can create sentences according to sentences size i give it <br>




7. Foldedr of Spinning <br>
8. Folder of Spam <br>
Kaggle: https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification <br>
   In this folder i practice with a the algoritme MultinomialNB from sklearn, i've classified the email with a small dataset from kaggle.
      
   |  Graph  | Description |
   |---|---|
   |  ![image](https://github.com/user-attachments/assets/a1c65d65-d277-4cf9-9abc-0ddee63f44df) | This graph has the binary data to see how many emails are spam or ham|
   |  ![image](https://github.com/user-attachments/assets/4d2e55bc-a573-42f4-9484-a745723dab1d) | This is a Heat Map of train data  |
   |  ![image](https://github.com/user-attachments/assets/f791690c-8a0a-4462-a762-5b643da0aff1) | This is a Heat Map of test data  |
   |  ![image](https://github.com/user-attachments/assets/493b47c6-20aa-4578-a72f-742e7fa5d8dd) | This is a word cloud  |
 
 
   

