import math
import pandas as pd

from clip_up.clip_up_general_utils import capitalize_only_first_letter

all_options = ['A', 'B', 'C', 'D', 'E']

i_cannot_answer_str = "I cannot answer."

def is_none(value):
    if value is None:
        return True
    if pd.isna(value):
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        try:
            option_value = row[option]
        except KeyError:
            break
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def build_choices(item):
    ret = {}
    for ch in 'ABCDE':
        try:
            if is_none(item[ch]) is False:
                ret[ch] = item[ch]
        except KeyError:
            continue
    return ret


def get_question_string(question, hint, options, upd_setting, upd_type):
    full_question_str = question
    if not is_none(hint):
        full_question_str = hint + '\n' + full_question_str
    for option_char, option in zip(all_options[:len(options)], options):
        full_question_str = full_question_str + '\n' + option_char + '. ' + option

    if upd_setting == 'base':
        if upd_type in ['aad', 'iasd', 'ivqd']:
            full_question_str = full_question_str + '\n'
        elif upd_type in ['open_ended']:
            full_question_str = full_question_str + '\n' + "Answer the question using a single word or phrase."
        else:
            assert False
    elif upd_setting in ['option', 'original']:
        if upd_type in ['aad', 'iasd', 'ivqd']:
            full_question_str = full_question_str + '\n' + "Answer with the option's letter from the given choices directly."
        else:
            assert False
    elif upd_setting == 'inst':
        if upd_type == 'ivqd':
            full_question_str = full_question_str + '\n' + "If the given image is irrelevant to the question, answer \"F. The image and question are irrelevant.\"."
        elif upd_type in ['aad', 'iasd']:
            full_question_str = full_question_str + '\n' + "If all the options are incorrect, answer \"F. None of the above\"."
        elif upd_type in ['open_ended']:
            full_question_str = full_question_str + '\n' + "When the provided information is insufficient, respond with 'I cannot answer.'. Answer the question using a single word or phrase."
        else:
            assert False
    else:
        assert False

    return full_question_str


def get_target_str(gt_answer, upd_type, item, eos_token):
    if upd_type in ['ivqd', 'aad', 'iasd']:
        if gt_answer == 'F' or gt_answer == i_cannot_answer_str:
            target_str = i_cannot_answer_str
        else:
            target_str = gt_answer + '. ' + item[gt_answer]
    elif upd_type in ['open_ended']:
        target_str = capitalize_only_first_letter(gt_answer)
    else:
        assert False

    if eos_token is not None:
        target_str += eos_token

    return target_str


def get_circular_item_index(item_index):
    circular_item_index = item_index % int(1e6)
    return circular_item_index
