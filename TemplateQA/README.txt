实现架构：
1. 读取数据集，数据集共三个文件，训练集，交叉验证集.dev和测试集，文件中每一行包含两个元素，字和标识。每一句话间由一个空格隔开。
2. 处理数据集
    1） 更新数据集中的标签，如： 单独的B-LOC→S-LOC；B-LOC,I-LOC→B-LOC,E-LOC；B-LOC,I-LOC,I-LOC→B-LOC, I-LOC, E-LOC
    2） 给每个char和tag分配一个id，得到一个包含所有字的字典dict，以及char_to_id, id_to_char, tag_to_id, id_to_tag, 将其存在map.pkl中
3. 准备训练集
        将训练集中的每句话变成4个list，第一个list是字，如[今，天，去，北，京]，第二个list是char_to_id [3,5,6,8,9]，第三个list是通过jieba分词得到的分词信息特征，如[1,3,0,1,3] （1，词的开始，2，词的中间，3，词的结尾，0，单个词），第四个list是target，如[0,0,0,2,3](非0的元素对应着tag_to_id中的数值)
4. BatchManager 将训练集划分成若干个batch，每个batch有20个句子，划分时，是现按句子长度从大到小排列
5. 配置model的参数
6. 构建模型
    1）input： 输入两个特征，char_to_id的list以及通过jieba得到的分词特征list
    2）embedding: 预先训练好了100维词向量模型，通过查询将得到每个字的100维向量，加上分词特征向量，输出到drouput(0.5)
    3）bi-lstm
    4）project_layer：两层参数（W,b），获得未归一化的矩阵
    5）loss_layer：内嵌了CRF