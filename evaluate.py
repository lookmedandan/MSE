def eval_sentence(y_pred, words, evltag=['E', 'S']):
    seg_pred = []
    word_pred = ''
    for i in range(len(y_pred)):
        word_pred += words[i]
        if y_pred[i] in evltag:
            seg_pred.append(word_pred)
            word_pred = ''
    seg_pred_str = ' '.join(seg_pred)
    return seg_pred_str


def cws_evaluate_word_PRF(y_pred, y, evltag=['E', 'S']):
    cor_num = 0
    yp_wordnum = sum([y_pred.count(c) for c in evltag])
    yt_wordnum = sum([y.count(c) for c in evltag])
    start = 0
    for i in range(len(y)):
        if y[i] in evltag:
            flag = True
            for j in range(start, i+1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i+1

    P = cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    R = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    F = 2 * P * R / (P + R)
    return P, R, F


def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id, evltag=['E', 'S']):
    cor_num = 0
    yt_wordnum = 0
    for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
        start = 0
        for i in range(len(y)):
            if y[i] in evltag:
                word = ''.join(sentence[start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y[j] != y_pred[j]:
                        flag = False
                if flag:
                    cor_num += 1
                start = i + 1

    OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    return OOV


def get_NER_scores(golden_lists, predict_lists):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0

    for idx in range(0,sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        gold_matrix = get_ner_BMES(golden_list)
        pred_matrix = get_ner_BMES(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)

    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure, (golden_num, predict_num, right_num)


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        
        else:
            continue

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)

    return stand_matrix
