from TemplateQA.main import evaluate_line


def align_data(data):
    data_aligned = {}
    for key, seq in data.items():
        str_aligned = ""
        if key == "entities":
            lenth = len(seq)
            for i in seq:
                lenth -= 1
                for j in i:
                    str_aligned += str(j) + " "
                if lenth != 0:
                    str_aligned += "; "
            data_aligned[key] = str_aligned
        else:
            data_aligned[key] = seq
    return data_aligned


def get_model_api(input):
    preds = evaluate_line(input)
    output_data = align_data(preds)
    return output_data


if __name__ == '__main__':
    a = get_model_api("章子怡演过卧虎藏龙吗?")
    print(a)