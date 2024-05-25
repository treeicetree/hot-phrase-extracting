# -*- encoding=utf-8 -*-

# 导入相关包
import datetime
import wsgiref.validate
import numpy as np
import pandas as pd
from collections import defaultdict
import thulac
import re
import joblib
from gensim.models import Word2Vec
import jieba.analyse
import jieba.posseg as pseg
import time
import jieba
import csv
import datetime
import time
import psutil
import os

# 导入相关库
from collections import Counter
from pprint import pprint

import MRR_marks

# 全局变量
globalContent = ""
targetNums = 100


# TextRank图权重计算部分
class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, start, end, weight=1):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self, iteration=10):
        """
        textrank算法的实现
        :param iteration: 迭代次数
        :return: dict type
        """
        print("begin to run rank func...")
        ws = defaultdict(float)
        outSum = defaultdict(float)  # 节点出度之和
        wsdef = 1.0 / (len(self.graph) or 1.0)  # 节点权值初始定义
        for n, edge in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((i[2] for i in edge), 0.0)
        sorted_keys = sorted(self.graph.keys())
        for i in range(iteration):  # 迭代
            # print("iteration %d..." % i)
            for n in sorted_keys:
                s = 0
                # 遍历节点的每条边
                for edge in self.graph[n]:
                    s += edge[2] / outSum[edge[1]] * ws[edge[1]]
                ws[n] = (1-self.d) + self.d*s  # 更新节点权值

        min_rank, max_rank = min(ws.values()), max(ws.values())
        # 归一化权值
        for n, w in ws.items():
            ws[n] = (w-min_rank/10) / (max_rank-min_rank/10)
        return ws

# TextRank算法
class TextRank:
    def __init__(self, data):
        """
        :param data: 输入的数据，字符串格式
        """
        self.data = data  # 字符串格式

    def extract_key_words(self,topK=20, window=4, iteration=200, allowPOS=('ns', 'n'), stopwords=True):
        """
        抽取关键词
        :param allowpos: 词性
        :param topK:   前K个关键词
        :param window: 窗口大小
        :param iteration: 迭代次数
        :param stopwords: 是否过滤停止词
        :return:
        """
        #短语部分
        textBefore = self.generate_word_list(allowPOS, stopwords)
        text = combine(textBefore)
        # print(text)
        # print(text)
        # #词组部分
        # text = self.generate_word_list(allowPOS, stopwords)
        #print(text)
        #text = extractQuotation()
        graph = UndirectWeightedGraph()
        # 定义共现词典
        cm = defaultdict(int)
        # 构建无向有权图
        for i in range(1, window):
            if i < len(text):
                text2 = text[i:]
                for w1, w2 in zip(text, text2):
                    cm[(w1, w2)] += 1
        numEdge = 0
        for terms, w in cm.items():
            graph.add_edge(terms[0], terms[1], w)
            numEdge += 1
            # print(terms[0],terms[1],w)

        print(numEdge)

        joblib.dump(graph, 'data/graph')
        ws = graph.rank(iteration)
        # for item in ws:
        #     print(item)
        #print(graph)
        return sorted(ws.items(), key=lambda x: x[1], reverse=True)[:topK]

    def generate_word_list(self, allowPOS, stopwords):
        """
        对输入的数据进行处理，得到分词及过滤后的词列表
        :param allowPOS: 允许留下的词性
        :param stopwords: 是否过滤停用词
        :return:
        """
        s = time.time()
        # thu_tokenizer = thulac.thulac(filt=True, rm_space=True, seg_only=False)
        # text = thu_tokenizer.cut(self.data)
        text = [(w.word, w.flag) for w in pseg.cut(self.data)]  # 词性标注
        word_list = []
        if stopwords:
            stop_words = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
            stopwords_news = [line.strip() for line in open('stopwords_news.txt', encoding='UTF-8').readlines()]
            all_stopwords = set(stop_words + stopwords_news)
        # 词过滤
        if text:
            for t in text:
                if len(t[0]) < 2:
                    continue
                if len(t[0]) < 2 or t[1] not in allowPOS:
                    continue
                if stopwords:
                    # 停用词过滤
                    if t[0] in all_stopwords:
                        continue
                word_list.append(t[0])
        return word_list

    def extract_key_sentences(self, topK=3, window=3, ndim=20, allowPOS=('ns', 'ni', 'nl'), stopwords=True, iteration=300):
        """
        抽取关键句子
        :param topK: 前K句话
        :param window: 窗口大小
        :param ndim: 词向量维度
        :param allowPOS: 词性
        :param iteration: 迭代次数
        :param stopwords: 是否过滤停用词
        :return:
        """
        try:
            text = joblib.load("data/sentence_vectors")
        except FileNotFoundError:
            text = self.bulid_sentence_vec(ndim, allowPOS, stopwords)
        graph = UndirectWeightedGraph()
        # 构建无向有权图
        for i in range(1, window):
            if i < len(text):
                text2 = text[i:]
                for w1, w2 in zip(text, text2):
                    if not np.isnan(self.cos_sim(w1[1], w2[1])):
                        graph.add_edge(w1[0], w2[0], self.cos_sim(w1[1], w2[1]))
        ws = graph.rank(iteration)
        s = list(ws.keys())
        topK_sentences = sorted(ws.items(), key=lambda x: x[1], reverse=True)[:topK]
        s_w_index_list = [[i, s.index(i[0])] for i in topK_sentences]
        res = sorted(s_w_index_list, key=lambda x: x[1])
        return sorted(res, key=lambda x: x[1])

    def bulid_sentence_vec(self, ndim, allowPOS, stopwords):
        """
        构建句向量
        :param ndim: 词向量维度
        :param allowPOS: 词性
        :param stopwords: 是否过滤停用词
        :return:
        """
        print("bulid_sentence_vec")
        try:
            self.sentence_list = joblib.load("data/sentence_list")
            model = Word2Vec.load("model/w2v_model")
        except FileNotFoundError:
            self.sentence_list = self.generate_sentence_list(allowPOS, stopwords)
            model = self.bulid_w2c(ndim)
        sentence_vectors = [[sentence[0][0], self.sentence_vec(model, sentence[1], ndim)] for sentence in self.sentence_list]
        joblib.dump(sentence_vectors, "data/sentence_vectors")
        return sentence_vectors

    def sentence_vec(self, model, sentence, ndim):
        vec = np.zeros(ndim)
        count = 0
        for word in sentence:
            try:
                vec += model.wv[word]
                count += 1
            except KeyError as e:
                continue
        if count != 0:
            vec /= count
        return vec

    def bulid_w2c(self, ndim):
        """
        训练Wordvec模型
        :param ndim:  词向量维度
        :return:
        """
        print("train bulid_w2c...")
        data = [s[1] for s in self.sentence_list]
        model = Word2Vec(data, size=ndim, window=3, iter=10)
        model.save("model/w2v_model")
        return model

    def generate_sentence_list(self, allowPOS, stopwords):
        """
        对输入的数据进行处理，得到句子列表（包含原句和分词列表）
        :param stopwords: 是否过滤停用词
        :return:
        """
        sentence_list = [[i] for i in re.split(r"[.。?!！？]", self.data)]  # 分句
        # thu_tokenizer = thulac.thulac(rm_space=True)
        new_sentence_list = []
        if stopwords:
            stop_words = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
            try:
                stopwords_news = [line.strip() for line in open('stopwords_news.txt', encoding='UTF-8').readlines()]
                all_stopwords = stop_words + stopwords_news
            except:
                all_stopwords = stop_words
        else:
            all_stopwords = ''
        for s in sentence_list:
            # word_list = thu_tokenizer.cut(s[0])  # 分词
            word_list = [(w.word, w.flag) for w in pseg.cut(self.data)]  # 词性标注
            new_word_list = []
            # 过滤
            if word_list:
                for w in word_list:
                    if allowPOS and w[1] not in allowPOS:
                            continue
                    if stopwords and w[0] in all_stopwords:
                            continue
                    new_word_list.append(w[0])
            if new_word_list:
                new_sentence_list.append([s, new_word_list])
        return new_sentence_list

    @classmethod
    def cos_sim(cls, vec_a, vec_b):
        """
        计算两个向量的余弦相似度
        :param vec_a:
        :param vec_b:
        :return:
        """
        vector_a = np.mat(vec_a)
        vector_b = np.mat(vec_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        if cos == 'nan':
            print(cos)
        sim = 0.5 + 0.5 * cos
        return sim

# 牛顿冷却法
def newtonCD():
    pass

# 打开csv文件通用函数
def openfile():
    with open('news.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

# 将 根据分割出来的词组进行组合
def combine(wordList):
    listCombine = []
    listRet = []
    for i in range(len(wordList)-1):
        if wordList[i] != wordList[i+1]:
            strCombine = str(wordList[i])+str(wordList[i+1])
        listCombine.append(strCombine)

    #print(listCombine)
    #print(Counter(listCombine))
    counterCombine = Counter(listCombine)
    for item in counterCombine.most_common(1000):
        #print(item)
        ##加入有计数的整体
        listRet.append(item)
        #print(item[0])
        #listRet.append(item[0])
    return listRet

# 提取引号内的内容
def extractQuotation():
    listContainer = []
    strContainer = ""

    startSet = False
    for item in globalContent:
        if startSet == True and item == "”":
            startSet = False
            if len(strContainer) > 1  and len(strContainer) < 10 and strContainer not in listContainer  :
                listContainer.append(strContainer)
            strContainer = ""
            continue
        if startSet == True:
            strContainer += str(item)
            continue
        if item == "“":
            startSet = True

    return listContainer

# 判断 词组是否属于 热词短语中的一部分
def getSameSeries(str1,str2):
    # str1, str2 = (str2, str1) if len(str1) > len(str2) else (str1, str2)
    # f = []
    # for i in range(len(str1), 0, -1):
    #     for j in range(len(str1) + 1 - i):
    #         e = str1[j:j + i]
    #         if e in str2:
    #             f.append(e)
    #
    #     if f:
    #         break
    # f1 = ",".join(f)
    # if(len(f1)!=0):
        return True
    # else:
    #     return False
    #print(f1)

if __name__ == '__main__':
    listCsv = openfile()
    #for i in range(1):
    #sentence = listCsv[4455][0]
    content = ""

    # ret = []
    for i in range(targetNums):
        content += (listCsv[0+i][0])

    #content = "新冠肺炎疫情发生以来，新冠病毒知识科普、居民防疫指南等信息牵动着所有人的神经。新冠肺炎是新型流感病毒，需要做好疫情防控工作。"
    globalContent = content
    #print(content)
    # myRet = extractSemiColon(content)
    # for item in myRet:
    #     print(item)

    # keywords = jieba.analyse.extract_tags(globalContent, topK=50, withWeight=False, allowPOS=())
    # for item in keywords:
    #     print(item)

    #TextRank算法调用
    tr = TextRank(content)
    '''
    key_sentences = tr.extract_key_sentences(topK=4, window=3, ndim=10)
    #print("myTR",key_sentences)
    key_words = jieba.analyse.textrank(content, topK=5, withWeight=True, allowPOS=('n', 'ni', 'nz'))
    #print("jiebaTR",key_words)
    key_words = jieba.analyse.extract_tags(content, topK=5, withWeight=True, allowPOS=('n', 'ni', 'nz'))   # tf-idf
    #print("jiebaExtract",key_words)
    key_words = tr.extract_key_words(topK=100,  window=5, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))
    '''
    #使用TextRank算法进行关键词提取
    ########短语部分#########
    # initTime = time.time()
    # key_words = tr.extract_key_words(topK=50, window=3, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))
    # endTime = time.time()
    # useTime = endTime - initTime
    # print("短语时间：",useTime)
    ########词组部分#########
    # initTime = time.time()
    key_words = tr.extract_key_words(topK=50, window=3, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))
    # endTime = time.time()
    # useTime = endTime - initTime
    # print("词组时间：",useTime)

    #牛顿热冷却算法部分
    countHalf1 = 0
    countHalf2 = 0

    contentHafl1 = ""
    contentHafl2 = ""

    mylistPhrase = []
    for item in key_words:
        #print(item[0][0])
        if item[0][1]>= 10 and len(mylistPhrase)<50:
            #print(item)
            #listPhrase.append(item[0][0])
            mylistPhrase.append(item)

    #print(mylistPhrase)

    listCsv = openfile()
    for i in range(1000):
        mystr = listCsv[i][1]
        splitStr = mystr.split('/')
        intStr = int(splitStr[1])
        if intStr<6:
            countHalf1 += 1
            contentHafl1 += (listCsv[i][0])
        else:
            countHalf2 += 1
            contentHafl2 += (listCsv[i][0])
    print("上半年条目数：",countHalf1)
    print("下半年条目数：",countHalf2)

    trHafl1 = TextRank(contentHafl1)
    trHafl2 = TextRank(contentHafl2)

    key_wordsHaf1 = trHafl1.extract_key_words(topK=200, window=3, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))

    key_wordsHaf2 = trHafl2.extract_key_words(topK=200, window=3, iteration=50, stopwords=True, allowPOS=('n', 'ni', 'nz'))
    wordsHalf1 = trHafl2.generate_word_list(allowPOS=('n', 'ni', 'nz'), stopwords=True, )
    counterHalf1 = Counter(wordsHalf1)

    wordsHalf2 = trHafl2.generate_word_list(allowPOS=('n', 'ni', 'nz'), stopwords=True, )
    counterHalf2 = Counter(wordsHalf2)

    halfPhrase1 = open("/Users/treeicetree/Desktop/热词提取/TextRank-master/halfFrequencyPhrase1.csv", "w+")
    for item in key_wordsHaf1:
        halfPhrase1.write(item[0][0])
        halfPhrase1.write('	')
        halfPhrase1.write(str(item[0][1]))
        halfPhrase1.write("\n")

    # halfPhrase1.close()
    # # halfPhrase1 = open("/Users/treeicetree/Downloads/TextRank-master/halfFrequencyPhrase1.txt", "w")
    listHalfPhrase1 = [list(item) for item in halfPhrase1]
    halfPhrase1.close()

    npHalfPhrase1 = np.array(listHalfPhrase1)
    list1 = []
    list2 = []
    for item in key_wordsHaf1:
        list1.append(item[0][0])
        list2.append(str(item[0][1]))

    a1 = [x for x in list1]
    b1 = [x for x in list2]

    # a1 = [x for x in npHalfPhrase1[:,0]]
    # b1 = [x for x in npHalfPhrase1[:,1]]
    dataframe = pd.DataFrame({'词语': a1, '热度': b1})
    dataframe.to_csv(r"/Users/treeicetree/Desktop/热词提取/TextRank-master/halfCSV1.csv", index=False)

    #####
    list3 = []
    list4 = []
    for item in key_wordsHaf2:
        list3.append(item[0][0])
        list4.append(str(item[0][1]))

    a2 = [x for x in list3]
    b2 = [x for x in list4]
    dataframe = pd.DataFrame({'词语': a2, '热度': b2})
    dataframe.to_csv(r"/Users/treeicetree/Desktop/热词提取/TextRank-master/halfCSV2.csv", index=False)
    #
    listH = []
    listHAll = []
    for index1 in range(len(a1)):
        for index2 in range(len(a2)):
            if(a1[index1] == a2[index2]):
                #print(a1[index1],":",index1,":",index2)
                #热冷却计算公式
                element = ((int(b1[index1])/countHalf1))/((int(b2[index2])/countHalf2))
                #180天 代指半年
                H = np.log(element) / 180
                listH = [a1[index1],H]
                listHAll.append(listH)
                #print(a1[index1],":",H)

    # listHAll.sort(key=lambda x: x[1],reverse=True)
    # print(listHAll)
    #
    Hfile = open("/Users/treeicetree/Desktop/热词提取/TextRank-master/HPhrasePoint.csv", "w")
    for itemHfile in listHAll:
        Hfile.write(itemHfile[0])
        #print(itemHfile[0])
        Hfile.write(str(itemHfile[1]))
        #print(itemHfile[1])
        Hfile.write("\n")

    Hfile.close()
    #############
    #content1 = ""今天小米双11可谓全家集体出动，除了手机，其他各条产品线也是一路狂奔，不断亮出耀眼数字刷屏。比如已经成为国内第一的小米电视，12个小时全渠道支付金额突破了10亿元，在天猫、京东、苏宁的销量、销售额全部都是第一，32寸、40寸、43寸、49寸、50寸、55寸、65寸七个单品额度也是销量第一。今天凌晨，小米电视更是只用9分02秒就入账1亿元，1小时58分到手5亿元。其他产品，截至中午12点50分，小米净水器全渠道支付金额破1亿元，创历史新高。截至16点20分，米家扫地机器人全渠道销售数量突破10万台，线下小米之家销量同比增长148倍，同时还是智能硬件单品单天最快破亿的。截至15点，天猫平台智能手环类目，小米手环3单品销量、销售额双第一。截止16点45分，天猫平台智能出行类目，九号平衡车单品销量、销售额双第一。手机方面，截至16点，小米MIX 3天猫、京东3000-4000元价位段销量第一，小米8天猫、京东、苏宁2000-3000元价位段销量第一，小米8青春版天猫、苏宁1000-2000元价位段销量第一。'''
    #content = "新华社武汉2月28日电（记者李思远、王作葵）中央赴湖北指导组成员、“床等人”现象，国家卫生健康委员会主任马晓伟28日说，武汉的新冠肺炎患者每4人就有1人是在方舱医院治疗的，方舱医院做到了“零感染、零死亡、零回头”。　　当日，国务院新闻办公室在湖北武汉举行新闻发布会，马晓伟等介绍了中央指导组指导疫情防控和医疗救治工作进展等方面的情况。　　马晓伟说，疫情发生以来，病人就医数量呈“井喷式”增长，大量病人在社区和社会流动，医疗资源紧张，床位不能满足应收尽收的要求，面临着延误治疗时机、造成疫情扩散的双重压力。在这种复杂情况下，中央指导组深入一线，果断作出建设方舱医院的决定，要求武汉市立即将一批体育场馆、会展中心逐步改造为方舱医院。　　武汉市已经建成16家方舱医院，实际开放床位13000多张，累计收治患者12000多人。目前，方舱还有7600多名患者，空余床位5600张，实现了“床等人”。　　马晓伟表示，建设方舱医院是一项非常关键、意义重大的举措，在短期内迅速扩充了医疗资源，解决了大量患者入院治疗的问题，避免了疫情以更快的速度扩散。方舱医院的建设，在防与治两个方面发挥了重要的、不可替代的作用，也为今后应对突发公共卫生事件、应对重大灾情疫情、迅速组织扩充医疗资源创造了一种新的模式。　　马晓伟说，方舱医院的大规模使用，在我国医学救援史上具有标志性意义。目前，方舱医院运行平稳，医患关系和谐，不少方舱医院还建立了医患临时党支部，开展了许多有利于康复的活动。可以说，方舱是名副其实的生命之舱。（来源：新华网，2020年02月28日）"  #print(content)


    #得到组合短语
    # word_list = tr.generate_word_list(allowPOS=('n', 'ni', 'nz'),stopwords=True)
    # #print(word_list)
    # listCombineRet = combine(word_list)
    #
    # loop = 0;
    # for item in listCombineRet:
    #     if(item[1]>10 and loop<50):
    #         loop += 1
    #         print(item)





    # for item in key_words:
    #     print(item[0][0])
    # pOutput = open("/Users/treeicetree/Downloads/TextRank-master/phraseOutput.txt", "w")
    # for item in key_words:
    #     pOutput.write(item[0][0])
    #     pOutput.write(str(item[1]))
    #     pOutput.write("\n")
    # pOutput.close()
    #

    ###################
    ##     计算得分   ##
    ###################
    retlistFinally  = MRR_marks.getKeyWordsNumber()
    # for item in retlistFinally:
    #     print(item)
    TotalMarks1 = 0

    combineCSV = open("/Users/treeicetree/Downloads/TextRank-master/CombinePhrase2.csv", "w")
    for item in key_words:
        combineCSV.write(item[0][0])
        combineCSV.write("\n")
    combineCSV.close()

    combineOpen = MRR_marks.openfileOFcombine()

    ifContains = False

    #词组
    ###############词组部分计算MRR得分过程##############
    for item in key_words:
        print(item[0])

    for j in range(len(retlistFinally)):
        for i in range(len(key_words)):
            # if int(combineOpen[i][1]) == 1:
            #     #ifContains = getSameSeries(combineOpen[i][0],itemlist)
            #ifContains = MRR_marks.CheckSame(key_words[0][0],itemlist) #短语部分
            ifContains = getSameSeries(key_words[i][0],retlistFinally[j][0]) #词组部分
            i1 = int(i / 5)
            if key_words[i][0][0] in retlistFinally :
                TotalMarks1 += 1/(i1+1)*retlistFinally[j][1]
            elif ifContains:
                TotalMarks1 +=( 1/(i1+1))/2*retlistFinally[j][1]
                pass
    TotalMarks1 /= targetNums
    print("MRR得分：" , TotalMarks1)

    ##打印内存
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    total_memory = psutil.virtual_memory().total

    # 将字节转换为 MB，并格式化输出
    total_memory_mb = total_memory / 1024 ** 2
    print(f"Total physical memory: {total_memory_mb:.2f} MB")

    print(f"RSS: {memory_info.rss / 1024 ** 2:.2f} MB")  # 常驻内存集大小
    print(f"VMS: {memory_info.vms / 1024 ** 2:.2f} MB")  # 虚拟内存大小
    ###############词组部分计算MRR得分过程##############

    #短语
    ###############短语部分计算MRR得分过程##############
    # tmpBool = True
    # listPhrase = []
    #
    # for item in key_words:
    #     #print(item[0][0])
    #     if item[0][1]>= 10 and len(listPhrase)<50:
    #           print(item[0][0])
    #         #listPhrase.append(item[0][0])
    #           listPhrase.append(item)
    #
    # for item in listPhrase:
    #     print(item[0][0])

    #
    #
    # for j in range(len(retlistFinally)):
    #     for i in range(len(key_words)):
    #         ifContains = MRR_marks.CheckSame(key_words[i][0][0],retlistFinally[j][0]) #短语部分
    #         #ifContains = getSameSeries(key_words[i][0],itemlist) #词组部分
    #         i1 = int(i/5)
    #         if key_words[i][0][0] == retlistFinally[j][0] :
    #             TotalMarks1 += 1/(i1+1)*retlistFinally[j][1]
    #         elif ifContains:
    #             TotalMarks1 += ( 1/(i1+1))/2*retlistFinally[j][1]
    #             pass
    # TotalMarks1 /= targetNums
    # print("MRR得分：" , TotalMarks1)
    ###############短语部分计算MRR得分过程##############


    ##打印内存
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    total_memory = psutil.virtual_memory().total

    # 将字节转换为 MB，并格式化输出
    total_memory_mb = total_memory / 1024 ** 2
    print(f"Total physical memory: {total_memory_mb:.2f} MB")

    print(f"RSS: {memory_info.rss / 1024 ** 2:.2f} MB")  # 常驻内存集大小
    print(f"VMS: {memory_info.vms / 1024 ** 2:.2f} MB")  # 虚拟内存大小

    ##    本实验部分   ##
    ###################
    #
    # qOutput = open("/Users/treeicetree/Downloads/TextRank-master/quotationOutput.txt", "w")
    # for item in key_words:
    #     qOutput.write(item[0])
    #     qOutput.write(str(item[1]))
    #     qOutput.write("\n")
    # qOutput.close()
    ''' 
    print("myTR keywords",key_words)
    for word in key_words:
        print(word[0])
        print(word[1])
        #print("\n")

    f = open("/Users/treeicetree/Downloads/TextRank-master/outPut.txt", "w")
    # for i in range(10):
    # f.write(retInfo)
    # f.write("\n")
    for item in key_words:
        f.write(item[0])
        f.write("\n")
    f.close()

    words = tr.generate_word_list(allowPOS=('n', 'ni', 'nz'),stopwords=True,)
    counter = Counter(words)


    # 打印前十高频词
    print("高频词")
    #pprint(counter.most_common(100))
    s = open("/Users/treeicetree/Downloads/TextRank-master/wordsFrequency.txt", "w")
    for item1 in counter.most_common(100):
        s.write(item1[0])
        s.write(str(item1[1]))
        s.write("\n")
    s.close()
    '''

#####################
### 打印组合词到CSV ###
#####################


