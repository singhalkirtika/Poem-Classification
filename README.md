# Poem-Classification
### DataSet ###: The dataset is taken from kaggle
(https://www.kaggle.com/ultrajack/modern-renaissance-poetry). There are 573 poems with three different genres: Love (326 poems), 
Nature (188 poems), Mythology and Folklore (59 poems).
### PreProcessin g### : The poems are processed for:
1. Removal of ‘\r\n’ using a regex expression
2. Using contractions function for expanding contractions like you’ve → you have, he’s → he is, aren’t → are not and similarly handle other contractions
3. Removing punctuations
4. Converting to Lower Case
5. Removal of stopwords like a, an, in, he
6. Lemmatization like better → good, playing → play
###Multinomial Naive Bayes model###
###LSTM model using Glove Word embeddings###
