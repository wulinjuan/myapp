# encoding=utf-8
from random import sample


def person_entity_to_sentence(entity, file):
    """
    把演员名字实体加入句子中，生成命名实体识别的训练集、验证集和测试集
    """
    q1 = str(entity) + "演了什么电影?"
    deal_1(q1, entity, file)

    q2 = str(entity) + "出演了哪些类型的电影？"
    deal_1(q2, entity, file)

    q3 = str(entity) + "共出演了几部电影？"
    deal_1(q3, entity, file)

    q4 = str(entity) + "出演过几部喜剧电影？"
    deal_1(q4, entity, file)

    q4 = str(entity) + "演过喜剧电影吗？"
    deal_1(q4, entity, file)

    q5 = str(entity) + "的出生日期。"
    deal_1(q5, entity, file)

    q6 = str(entity) + "出生在哪？"
    deal_1(q6, entity, file)

    q7 = str(entity) + "的英文名是什么？"
    deal_1(q7, entity, file)
    return


def movie_entity_to_sentence(entity, file):
    """
    把电影名称实体加入句子中，生成命名实体识别的训练集、验证集和测试集
    """

    q1 = "哪些演员出演了"
    deal_2(q1, entity, file)

    q2 = str(entity) + "的上映日期。"
    deal_3(q2, entity, file)

    q3 = str(entity) + "的评分。"
    deal_3(q3, entity, file)

    q4 = str(entity) + "是什么类型的电影？"
    deal_3(q4, entity, file)

    q4 = str(entity) + "是喜剧电影吗？"
    deal_3(q4, entity, file)
    return


def person_s_to_sentence(person_s, file):
    """
    把两个演员实体加入句子中，生成命名实体识别的训练集、验证集和测试集
    """
    j = 0
    for i in person_s:
        if j == 0:
            person = i.rstrip().split()[0]
            # entity = unicode(person, 'utf-8')
            k = 0
            for i in person:
                if k == 0:
                    file.writelines(i + u" B-PER" + '\n')
                    k = 1
                else:
                    file.writelines(i + u" I-PER" + '\n')
            j = 1
        else:
            file.writelines("和" + u" O" + '\n')
            person = i.rstrip().split()[0]
            # entity = unicode(person, 'utf-8')
            k = 0
            for i in person:
                if k == 0:
                    file.writelines(i + u" B-PER" + '\n')
                    k = 1
                else:
                    file.writelines(i + u" I-PER" + '\n')
    q = "共同出演了什么电影？"
    # q = unicode(q_1, 'utf-8')
    for i in q:
        file.writelines(i + u" O" + '\n')
    file.writelines('\n')
    return


def person_movie_to_sentence(person1, movie1, file):
    """
    把电影和演员实体加入句子中，生成命名实体识别的训练集、验证集和测试集
    """
    # entity_1 = unicode(person1, 'utf-8')
    k = 0
    for i in person1:
        if k == 0:
            file.writelines(i + u" B-PER" + '\n')
            k = 1
        else:
            file.writelines(i + u" I-PER" + '\n')
    q = "出演了"
    # q = unicode(q_1, 'utf-8')
    for i in q:
        file.writelines(i + u" O" + '\n')
    # entity_2 = unicode(movie1, 'utf-8')
    j = 0
    for i in movie1:
        if j == 0:
            file.writelines(i + u" B-MOV" + '\n')
            j = 1
        else:
            file.writelines(i + u" I-MOV" + '\n')
    file.writelines("吗" + u" O" + '\n')
    file.writelines("？" + u" O" + '\n')
    file.writelines('\n')
    return


def deal_1(q, entity, file):
    # entity = unicode(entity, 'utf-8')
    length = len(entity)
    # q = unicode(q, 'utf-8')
    j = 0
    for i in q:
        if j == 0:
            file.writelines(i + u" B-PER" + '\n')
            j += 1
        elif j < length:
            file.writelines(i + u" I-PER" + '\n')
            j += 1
        else:
            file.writelines(i + u" O" + '\n')
    file.writelines('\n')
    return


def deal_2(q, entity, file):
    # entity = unicode(entity, 'utf-8')
    # q = unicode(q, 'utf-8')
    j = 0
    for i in q:
        file.writelines(i + u" O" + '\n')
    for i in entity:
        if j == 0:
            file.writelines(i + u" B-MOV" + '\n')
            j = 1
        else:
            file.writelines(i + u" I-MOV" + '\n')
    file.writelines('?' + u" O" + '\n')
    file.writelines('\n')
    return


def deal_3(q, entity, file):
    # entity = unicode(entity, 'utf-8')
    length = len(entity)
    # q = unicode(q, 'utf-8')
    j = 0
    for i in q:
        if j == 0:
            file.writelines(i + u" B-MOV" + '\n')
            j += 1
        elif j < length:
            file.writelines(i + u" I-MOV" + '\n')
            j += 1
        else:
            file.writelines(i + u" O" + '\n')
    file.writelines('\n')
    return


def deal_sum():
    f_person = open('./external_dict/person_name.txt', 'r', encoding='UTF-8')
    f_movie = open('./external_dict/movie_title.txt', 'r', encoding='UTF-8')
    f_train = open('./data/train.txt', 'w',encoding='utf-8')
    f_dev = open('./data/dev.txt', 'w',encoding='utf-8')
    f_test = open('./data/test.txt', 'w',encoding='utf-8')

    i_person = f_person.readlines()
    i_movie = f_movie.readlines()

    for i in i_person[:174]:
        person = i.rstrip().split()[0]
        person_entity_to_sentence(person, f_train)

    for i in i_person[175:224]:
        person = i.rstrip().split()[0]
        person_entity_to_sentence(person, f_test)

    for i in i_person[225:254]:
        person = i.rstrip().split()[0]
        person_entity_to_sentence(person, f_dev)

    for i in i_movie[:1499]:
        movie = i.rstrip().split()[0]
        movie_entity_to_sentence(movie, f_train)

    for i in i_movie[1500:1999]:
        movie = i.rstrip().split()[0]
        movie_entity_to_sentence(movie, f_test)

    for i in i_movie[2000:2224]:
        movie = i.rstrip().split()[0]
        movie_entity_to_sentence(movie, f_train)

    for i in range(7000):
        person_s = sample(i_person[:254], 2)
        person_s_to_sentence(person_s, f_train)

    for i in range(2000):
        person_s = sample(i_person[:254], 2)
        person_s_to_sentence(person_s, f_test)

    for i in range(1000):
        person_s = sample(i_person[:254], 2)
        person_s_to_sentence(person_s, f_dev)

    for i in i_movie[:2224]:
        j = sample(i_person[:254], 1)
        person = j[0].rstrip().split()[0]
        movie = i.rstrip().split()[0]
        person_movie_to_sentence(person, movie, f_train)

    for i in i_movie[:2224]:
        j = sample(i_person[:254], 1)
        person = j[0].rstrip().split()[0]
        movie = i.rstrip().split()[0]
        person_movie_to_sentence(person, movie, f_test)

    for i in i_movie[:2224]:
        j = sample(i_person[:254], 1)
        person = j[0].rstrip().split()[0]
        movie = i.rstrip().split()[0]
        person_movie_to_sentence(person, movie, f_dev)

    f_train.close()
    f_test.close()
    f_movie.close()
    f_person.close()
    f_dev.close()


if __name__ == '__main__':
    deal_sum()
    f_train = open('./data/train.txt', 'r',encoding='utf-8')
    f_dev = open('./data/dev.txt', 'r',encoding='utf-8')
    f_test = open('./data/test.txt', 'r',encoding='utf-8')
    f_train2 = open('./data/example.train', 'r',encoding='utf-8')
    f_dev2 = open('./data/example.dev', 'r',encoding='utf-8')
    f_test2 = open('./data/example.test', 'r',encoding='utf-8')
    f_train1 = open('./data/train_set.txt', 'w',encoding='utf-8')
    f_dev1 = open('./data/dev_set.txt', 'w',encoding='utf-8')
    f_test1 = open('./data/test_set.txt', 'w',encoding='utf-8')

    for i in f_train.readlines():
        f_train1.writelines(i)

    for i in f_train2.readlines():
        f_train1.writelines(i)

    for i in f_test.readlines():
        f_test1.writelines(i)

    for i in f_test2.readlines():
        f_test1.writelines(i)

    for i in f_dev.readlines():
        f_dev1.writelines(i)

    for i in f_dev2.readlines():
        f_dev1.writelines(i)

    f_dev.close()
    f_dev1.close()
    f_test1.close()
    f_test.close()
    f_train.close()
    f_train1.close()
    f_dev2.close()
    f_test2.close()
    f_train2.close()