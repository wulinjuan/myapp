# encoding=utf-8
from TemplateQA import sparql_endpoint
from TemplateQA import question2sparql


def main(question):
    virtuoso = sparql_endpoint.Virtuoso()  # 链接Virtuoso数据库
    # 初始化自然语言到SPARQL查询的模块，参数是外部词典列表。
    q2s = question2sparql.Question2Sparql(['./external_dict/movie_title.txt', './external_dict/person_name.txt'])

    while True:
        # question = input("请输入问题：")  # 问题输入
        my_query = q2s.get_sparql(question)  # 根据模板匹配，将问题转化为SPARQL语句
        print(str(my_query))
        if my_query is not None:
            result = virtuoso.get_sparql_result(my_query)
            value = virtuoso.get_sparql_result_value(result)

            if isinstance(value, bool):  # 判断结果是否是布尔值，是布尔值则提问类型是"ASK"，回答“是”或者“不知道”
                if value is True:
                    return str(my_query), 'Yes'
                else:
                    return str(my_query), 'I don\'t know. :('
            else:
                # 查询结果为空，根据OWA，回答“不知道”
                if len(value) == 0:
                    return str(my_query), 'I don\'t know. :('
                elif len(value) == 1:
                    return str(my_query), str(value[0])
                else:
                    output = ''
                    for v in value:
                        output += v + u'、'
                        return str(my_query), str(output[0:-1])

        else:
            return str(my_query), 'I can\'t understand. :('


if __name__ == '__main__':
    main()