# -*- encoding=utf-8 -*-
import csv
import jieba
from collections import Counter

newsNumber = 100
listTotal = []
listTotalFinallyOutside = []

# 在标注数据中计算关键词出现的频率
def getKeyWordsNumber():
    listCSV = openfile()
    listTotalFinally = []
    listPharseAndCount = []
    for i in range(newsNumber):
        sentence = listCSV[i][2]
        a = sentence.split(" ")
        listTotal.extend(a)
    counter = Counter(listTotal)
    for item in counter:
        listTotalFinally.append(item)
    for item in counter.most_common(len(listTotalFinally)):
        #print(item)
        listPharseAndCount.append(item)

    return listPharseAndCount

# 打开csv文件通用函数
def openfile():
    with open('/Users/treeicetree/Desktop/热词提取/TextRank-master/labelKey.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

# 打开组合短语文件
def openfileOFcombine():
    with open('/Users/treeicetree/Desktop/热词提取/TextRank-master/CombinePhrase.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

# listTotalFinallyOutside = getKeyWordsNumber()
# print(listTotalFinallyOutside)

# 检查提取的短语组合是否与标注数据有一致的地方
def CheckSame(str1,str2):
    a_tokenizer= jieba.cut(str1)
    a_set = set()
    for item in a_tokenizer:
        a_set.add(item)

    b_tokenizer= jieba.cut(str2)
    b_set = set()
    for item in b_tokenizer:
        b_set.add(item)

    #print(a_set)
    #print(b_set)
    if len(a_set & b_set) > 0:
 	    return True
    else:
 	    return False

