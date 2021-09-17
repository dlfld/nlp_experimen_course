
class WordList:
    """
    获取词表的静态类
    """

    @staticmethod
    def generate_word_seg_list(sentences, split=' '):
        """
        生成词表，传入所有句子的列表，并且每一个列表中是分词之后的句子，可自定义分词之后句子的分隔符
        :param sentences:  句子列表
        :param split: 每一个句子分词之后的分隔符，默认为一个空格
        :return: 词表map 键是一个单词或字 值是他的索引
        """
        word_map = {}
        for sentence in sentences:
            words = sentence.split(split)
            for word in words:
                if word not in word_map:
                    word_map[word] = len(word_map)
        return word_map





