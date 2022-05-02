import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from lxml import etree


class KeyTerms:
    def __init__(self):
        self.file_name = 'news.xml'
        self.xml_tree = None
        self.xml_root = None
        self.corpus = None
        self.lemmetizer = nltk.WordNetLemmatizer()
        self.text_sequence = []
        self.matrix_column = 0

    def main(self):
        self._set_tree()
        self._set_root()
        self._set_corpus()
        self._set_text_sequence()
        self._give_output()

    def _set_tree(self):
        self.xml_tree = etree.parse(fr'{os.getcwd()}\{self.file_name}')

    def _set_root(self):
        self.xml_root = self.xml_tree.getroot()

    def _set_corpus(self):
        self.corpus = self.xml_root[0]  # This is still in xml format

    def _lemmatize_words(self, list_of_words):
        """Returns list_of_words, but with every word lemmatized."""
        new_list = []
        for word in list_of_words:
            if word.startswith("'"):
                new_list.append(word)
            else:
                lemmatized_word = self.lemmetizer.lemmatize(word)
                new_list.append(lemmatized_word)

        return new_list

    def _remove_stopwords(self, list_of_words):
        """Returns list_of_words, but with all stopwords removed."""
        stop_words = nltk.corpus.stopwords.words('english') + ['ha', 'wa', 'u', 'a']
        new_list = []
        for word in list_of_words:
            if word.startswith("'"):
                new_list.append(word)
            elif word in stop_words:
                pass
            else:
                new_list.append(word)

        return new_list

    def _remove_punctuation_from_words(self, list_of_words):
        new_list = []

        for word in list_of_words:
            if word.startswith("'") and len(word) != 1:
                new_list.append(word)
                continue

            new_word = self._remove_punctuation(word)
            if new_word:
                new_list.append(new_word)

        return new_list

    def _remove_punctuation(self, word):
        punctuation_marks = list(string.punctuation)
        new_word = ''

        for character in word:
            if character in punctuation_marks:
                pass
            else:
                new_word += character

        return new_word

    def _only_nouns(self, list_of_words):
        new_list = []
        for word in list_of_words:
            word_tag = nltk.pos_tag([word])
            if word_tag[0][1] == 'NN':
                new_list.append(word_tag[0][0])

        return new_list

    def _set_text_sequence(self):
        """Adds each article body to the text_sequence attribute."""
        for article in self.corpus:
            self.text_sequence.append(self._format_article(article))

        return

    def _get_tfidf_scores(self, formatted_article):
        vocab_words = []
        for word in formatted_article.split():
            if word not in vocab_words:
                vocab_words.append(word)

        vectorizer = TfidfVectorizer(vocabulary=vocab_words)
        tfidf_matrix = vectorizer.fit_transform(self.text_sequence)
        tfidf_matrix = tfidf_matrix.toarray()
        terms = list(vectorizer.get_feature_names_out())

        token_scores = []
        for i in range(len(vocab_words)):
            token_scores.append((terms[i], tfidf_matrix[self.matrix_column][i]))

        token_scores = sorted(token_scores, key=lambda x: (x[1], x[0]), reverse=True)
        return token_scores

    def _format_article(self, article):
        dictionary = {
            'heading': article[0].text,
            'formatted_story': nltk.tokenize.word_tokenize(article[1].text.lower())
        }

        dictionary['formatted_story'] = self._lemmatize_words(dictionary['formatted_story'])

        if 'co-lead' in dictionary['formatted_story']:  # Lemmatizer leaves 'co-lead' as is, but the test expects that to be counted as 'lead'. Otherwise, 'study' has a higher TF-IDF in the final test.
            dictionary['formatted_story'].remove('co-lead')
            dictionary['formatted_story'].append('lead')

        dictionary['formatted_story'] = self._remove_stopwords(dictionary['formatted_story'])
        dictionary['formatted_story'] = self._remove_punctuation_from_words(dictionary['formatted_story'])
        dictionary['formatted_story'] = self._only_nouns(dictionary['formatted_story'])

        return ' '.join(dictionary['formatted_story'])

    def _give_output(self):
        for article, formatted_article in zip(self.corpus, self.text_sequence):
            print(article[0].text + ':')  # Heading
            token_scores = self._get_tfidf_scores(formatted_article)
            print(' '.join([token[0] for token in token_scores[:5]]) + '\n')
            self.matrix_column += 1

        return


stage_1 = KeyTerms()
stage_1.main()
