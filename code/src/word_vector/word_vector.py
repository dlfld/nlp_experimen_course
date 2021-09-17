import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class PrivateUtils:
    @staticmethod
    def init_default_map(sentences, split=' '):
        """
        初始化一个空的word_map key是单词 value是0
        :param sentences:
        :param split:
        :return:
        """
        word_map = {}
        for sentence in sentences:
            words = sentence.split(split)
            for word in words:
                if word not in word_map:
                    word_map[word] = 0
        return word_map



class WordVector:
    """
    获取词向量的工具类
    """
    # @staticmethod
    # def tfidf_transform(sentences):
    #     """
    #     使用tfidf的方法获取词向量
    #     :param sentences: 句子列表
    #     :return: 词向量
    #     """
    #     vectorizer = CountVectorizer(max_df=0.8, min_df=3)
    #     tfidftransformer = TfidfTransformer()
    #     res_tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(sentences))
    #     return res_tfidf
    @staticmethod
    def tfidf_transform(sentences, word_map, split=' '):
        """
        tfidf
        遍历每一个句子
            统计每个句子中每个单词在句子中出现的次数
            统计每个单词在这句话中出现的次数
            计算每个单词的tf和idf
        :param sentences:
        :param word_map: 语料库  key是单词 value是单词的索引
        :param split:
        :return:
        """
        tfidf_features = np.zeros((len(sentences), len(word_map)))
        # 语料库中文档总数
        total_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            # 获取单个单词
            words = sentence.strip().split(split)
            # 一句话中的总词数
            total_words = len(words)
            word_map = {}
            # 统计每个单词在这句话中出现的次数
            for word in words:
                if word in word_map:
                    word_map[word] += 1
                else:
                    word_map[word] = 1
            # 计算每个单词的tf和idf
            for word_key in word_map:
                word_tf = word_map[word_key] / total_words
                # 包含该词的文档数
                word_key_count = 0
                # 统计包含该词的文档数
                for single_sentence in sentences:
                    if word_key in single_sentence:
                        word_key_count += 1
                word_idf = np.log(total_sentences / word_key_count)
                tfidf_features[i][word_map[word_key]] = word_tf * word_idf
        return tfidf_features

    @staticmethod
    def word2vec_transform(sentences, word_map, split=' '):
        """
        word2vec_transform
        首先初始化一个和词表一样大小的0矩阵
        遍历给的句子，提取出一条句子中的每一个单词，和传入的词表做对比取出这个单词在词表中的位置。
        并在向量举证的位置上加一
        :param sentences:  句子
        :param word_map:   词表
        :param split: 一句话中分词的分隔符，默认为一个空格
        :return: 词向量
        """
        bag_of_word_feature = np.zeros((len(sentences), len(word_map)))
        for index, sentence in enumerate(sentences):
            words = sentence.split(split)
            for word in words:
                sentence_loc = word_map[word]
                bag_of_word_feature[index, sentence_loc] += 1
        return bag_of_word_feature

    @staticmethod
    def onehot_transform(sentences, word_map, split=' '):
        """
        onehot_transform
        首先初始化一个和词表一样大小的0矩阵
        遍历给的句子，提取出一条句子中的每一个单词，和传入的词表做对比取出这个单词在词表中的位置。
        并使得向量举证的位置上的值为1
        :param sentences:  句子
        :param word_map:   词表
        :param split: 一句话中分词的分隔符，默认为一个空格
        :return: 词向量
        """
        onehot_res = np.zeros((len(sentences), len(word_map)))
        for index, sentence in enumerate(sentences):
            words = sentence.split(split)
            for word in words:
                sentence_loc = word_map[word]
                onehot_res[index, sentence_loc] = 1
        return onehot_res
