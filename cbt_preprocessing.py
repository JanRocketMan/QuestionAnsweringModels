from numpy import array, empty, zeros
from random import sample, randint
from itertools import islice

def highlight(word):
    return '\x1b[31m' + word + '\x1b[0m'

class CBTProcessor:
    def __init__(self, filename, word2vecfile, embed_dim, term_lower_bound=10):
        # filename - path to train corpus, word2vecfile - path to word2vec txt file,
        # embed_dim - dimension of the embeddings stored in word2vecfile

        self.words = set()
        self.word_to_id, self.id_to_word = {}, {}
        self.max_doc_len, self.max_query_len = 0, 0
        self.__train_data, self.val_data, self.__test_data = {}, {}, {}
        self.train_data_len, self.val_data_len, self.test_data_len = 0, 0, 0
        self.embed_dim = embed_dim
        self.__pre_trained, self.__new_words = set(), set()
        
        word2vec_set = self.__get_word2vec_set(word2vecfile)
        self.__retrieve_dictionary(filename, word2vec_set, term_lower_bound)
        self.embeddings = zeros((len(self.words), self.embed_dim))
        self.__get_word2vec_embeddings(word2vecfile)

    def __get_word2vec_set(self, word2vecfile):
        # Extract all words from a word2vec file
        word2vec_set = set()
        with open(word2vecfile, 'r') as fin:
            for line in fin:
                word2vec_set.add(line.split(None, 1)[0])
        return word2vec_set

    def __retrieve_dictionary(self, filename, word2vec_set, term_lower_bound):
        # parse txt file, count word frequences
        tokens_freq = {}
        with open(filename, 'r') as fin:
            for line in fin:
                if line[0] == '_':
                    continue
                tokens = line.split(' ')
                if not tokens or tokens[0] == 'CHAPTER':
                    continue
                self.words |= set(tokens)
                for tok in tokens:
                    if tok in tokens_freq.keys():
                        tokens_freq[tok] += 1
                    else:
                        tokens_freq[tok] = 1
        
        # drop all rare words from the dictionary
        rare_words = set([tok for tok in self.words if 
                               tokens_freq[tok] < term_lower_bound and 
                               not tok in word2vec_set])
        self.words -= rare_words
        # delete all copies
        repeated_words = set([tok for tok in self.words if
                             tok.isalpha() and not tok.islower() and tok.lower() in self.words])
        self.words -= repeated_words
        # lowercast all words
        self.words = set([tok.lower() for tok in self.words])
        # get all pre-trained words
        self.__pre_trained = set([tok for tok in self.words if (tok in word2vec_set) or 
                           (tok.lower() in word2vec_set)])
        # add special symbols
        self.words |= {'XXXXX'}
        self.words |= {'<NA>'}
        self.__new_words = self.words - self.__pre_trained

        assert len(self.__pre_trained) + len(self.__new_words) == len(self.words)
        
        # make id-word dicts
        # first index pre-trained words, then the others
        temp = {word: i for i, word in enumerate(self.__pre_trained)}
        c_len = len(temp)
        self.word_to_id = temp.copy()
        self.word_to_id.update({word: (i+c_len) for i, word in enumerate(self.__new_words)})
        
        temp2 = {i: word for i, word in enumerate(self.__pre_trained)}
        self.id_to_word = temp2.copy()
        self.id_to_word.update({(i+c_len):word for i, word in enumerate(self.__new_words)})
        
        assert len(self.word_to_id) == len(self.id_to_word) == len(self.words)
        # show statistics
        print('Words extracted. Total number:', len(self.words))
        print('Number of pre-trained:', len(self.__pre_trained))

    def __get_word2vec_embeddings(self, word2vecfile):
        # retrive relevant embeddings
        with open(word2vecfile, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(' ')
                word = line_split[0]
                if word in self.__pre_trained:
                    vec = array(line_split[1:], dtype=float)
                    self.embeddings[self.word_to_id[word]] = vec
        assert len(self.embeddings) == len(self.words)

    def fit_on_texts(self, filename, filetype, max_doc_len=1000, max_query_len=150, query_tok='21'):
        # process txt to make (D, Q, A) triplets
        self.max_doc_len, self.max_query_len = max_doc_len, max_query_len
        with open(filename, 'r') as fin:
            cDoc = list()
            ind = 0
            for line in fin:
                tokens = line.replace('\n', '').split(' ')
                if tokens and tokens[0] == query_tok:
                    # make a query from 21-st sentence
                    temp = tokens[-1].split('\t')
                    temp.remove('')
                    cands = temp[-1].split('|')
                    
                    cQuery = [self.__process_token(tok) for tok in tokens[1:-1]+[temp[0]]]
                    cCands = [self.__process_token(tok) for tok in cands]
                    cAns = self.__process_token(temp[1])
                    if filetype == 'train':
                        self.__train_data.update({ind: self.__process_lists(cDoc, cQuery, cCands, cAns)})
                    elif filetype == 'val':
                        self.val_data.update({ind: self.__process_lists(cDoc, cQuery, cCands, cAns)})
                    elif filetype == 'test':
                        self.__test_data.update({ind: self.__process_lists(cDoc, cQuery, cCands, cAns)})
                    else:
                        raise TypeError('\'filetype\' should be \'train\', \'val\' or \'test\'')
                    ind += 1
                    cDoc = list()
                else:
                    cDoc += [self.__process_token(tok) for tok in tokens[1:]]
        if filetype == 'train':
        	self.train_data_len = ind
        elif filetype == 'val':
        	self.val_data_len = ind
        elif filetype == 'test':
        	self.test_data_len = ind

    def __process_token(self, tok):
        # convert token to index
        res = self.word_to_id['<NA>']
        if tok in self.words:
            res = self.word_to_id[tok]
        elif tok.lower() in self.words:
            res = self.word_to_id[tok.lower()]
        elif tok.upper() in self.words:
            res = self.word_to_id[tok.upper()]
        return res

    def __process_lists(self, cDoc, cQuery, cCands, cAns):
        # convert lists to triplets
        nDoc = empty(self.max_doc_len)
        if len(cDoc) > self.max_doc_len:
            # crop too long
            nDoc = array(cDoc[:self.max_doc_len])
        else:
            # pad too short
            nDoc = array(cDoc + [self.word_to_id['<NA>']] * (self.max_doc_len - len(cDoc)))
        # same for query text
        nQuery = empty(self.max_query_len)
        if len(cQuery) > self.max_query_len:
            nQuery = array(cQuery[:self.max_query_len])
        else:
            nQuery = array(cQuery + [self.word_to_id['<NA>']] * (self.max_query_len - len(cQuery)))
        return (nDoc, nQuery, cCands, cAns, len(cDoc), len(cQuery))

    def sample_batch(self, filetype, batch_size, offset=None):
        if filetype == 'train':
            data = self.__train_data
        elif filetype == 'val':
            data = self.val_data
        elif filetype == 'test':
            data = self.__test_data
        else:
            raise TypeError('filetype could be "train", "val" or "test"')
        
        if offset is None:
            inds = sample(range(len(data)), batch_size)
        else:
            inds = range(offset, offset + batch_size)
        sample_D, sample_Q, sample_C, sample_A, sample_mask = [],[],[],[],[]
        for ind in inds:
            newsample = data[ind]
            sample_D += [newsample[0]]
            sample_Q += [newsample[1]]
            sample_C += [newsample[2]]
            sample_A += [newsample[3]]
            temp_list = zeros((len(newsample[0]),len(newsample[1])))
            temp_list[0:newsample[-2],0:newsample[-1]] = 1
            sample_mask += [temp_list]
        return (array(sample_D), array(sample_Q), 
                array(sample_C), array(sample_A), array(sample_mask))

    def show_example(self, sample):
        # print one example from sample
        rand_ind = randint(0, len(sample[0])-1)
        rand_doc = [self.id_to_word[tok] for tok in sample[0][rand_ind] if tok != self.word_to_id['<NA>']]
        rand_query = [self.id_to_word[tok] for tok in sample[1][rand_ind] if tok != self.word_to_id['<NA>']]
        rand_ans = self.id_to_word[sample[-2][rand_ind]]
        
        rand_doc = [highlight(elem) if elem in self.__new_words else elem for elem in rand_doc]
        rand_query = [highlight(elem) if elem in self.__new_words else elem for elem in rand_query]
        
        print('DOC:')
        print(' '.join(rand_doc))
        print('-'*10)
        print('QUERY:')
        print(' '.join(rand_query))
        print('-'*10)
        print('ANSWER:')
        print(rand_ans)
