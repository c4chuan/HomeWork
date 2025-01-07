import random
from collections import defaultdict, Counter

class UnsmoothedNGram:
    def __init__(self, n):
        """
        初始化n元语法模型。
        :param n: n的值，表示n元语法的阶数（1为一元语法，2为二元语法）。
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)  # 存储n元语法的计数
        self.context_counts = Counter()          # 存储上下文的计数

    def train(self, corpus):
        """
        训练模型，计算n元语法的计数。
        :param corpus: 一个包含训练语料的列表，每个元素是一个句子字符串。
        """
        for sentence in corpus:
            tokens = sentence.split()
            tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]  # 添加开始和结束标记
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = tuple(tokens[i:i + self.n - 1])  # 上下文是ngram的前n-1个词
                self.ngram_counts[context][ngram[-1]] += 1
                self.context_counts[context] += 1

    def sentence_probability(self, sentence):
        """
        计算给定句子的概率。
        :param sentence: 输入的句子字符串。
        :return: 句子的概率。
        """
        tokens = sentence.split()
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        probability = 1.0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            count_ngram = self.ngram_counts[context][ngram[-1]]
            count_context = self.context_counts[context]
            if count_context > 0:
                probability *= count_ngram / count_context
            else:
                probability = 0.0
                break

        return probability

    def generate_sentence(self, max_length=20):
        """
        利用模型采样生成句子。
        :param max_length: 生成句子的最大长度。
        :return: 生成的句子字符串。
        """
        sentence = ["<s>"] * (self.n - 1)
        for _ in range(max_length):
            context = tuple(sentence[-(self.n - 1):])
            if context not in self.ngram_counts:
                break
            next_word = self._sample_next_word(context)
            if next_word == "</s>":
                break
            sentence.append(next_word)

        return " ".join(sentence[(self.n - 1):])

    def _sample_next_word(self, context):
        """
        根据上下文采样下一个词。
        :param context: 当前的上下文（元组形式）。
        :return: 采样得到的下一个词。
        """
        next_words = self.ngram_counts[context]
        total_count = sum(next_words.values())
        rand_val = random.uniform(0, total_count)
        cumulative = 0
        for word, count in next_words.items():
            cumulative += count
            if rand_val <= cumulative:
                return word
        return "</s>"

if __name__ == '__main__':
    corpus = [
        "I am Sam",
        "Sam I am",
        "I do not like green eggs and Sam"
    ]

    # 训练一元语法模型
    unigram_model = UnsmoothedNGram(1)
    unigram_model.train(corpus)

    # 训练二元语法模型
    bigram_model = UnsmoothedNGram(2)
    bigram_model.train(corpus)

    # 测试句子概率
    test_sentence = "I am Sam"
    print("对于测试句子:I am Sam")
    print("Unigram模型的预测概率:", unigram_model.sentence_probability(test_sentence))
    print("Bigram模型的预测概率:", bigram_model.sentence_probability(test_sentence))

    # 生成句子
    print("生成的句子(Bigram):", bigram_model.generate_sentence())
