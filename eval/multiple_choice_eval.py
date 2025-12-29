import json
import string
from collections import defaultdict

from prompting.prompting_utils import i_cannot_answer_str, is_none, get_circular_item_index
from eval.eval_utils import call_open_ai_gpt_3_5, print_dual_accuracy_scores, evaluate_dual_accuracy, \
    evaluate_circular_accuracy, print_regular_and_circular_accuracy_scores, get_eval_type_specific


# Function based on 'https://github.com/AtsuMiyai/UPD' code
def infer_option_by_string_matching(answer, option_dict, question_type=None, valid_option=None):
    if valid_option is None:
        valid_option = list(option_dict.keys())
        if question_type == 'inst':
            valid_option.append("F")

    if 'Failed to obtain answer via API' in answer:
        return False

    answer = answer.strip()

    ch_cand_list = []
    punctuations = [".", ")", ","]

    # Inside each 'if' below, we check that if the prediction matches to one of the choices by using simple string
    # patterns
    if "A" in valid_option:
        characters = ["B", "C", "D", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["A)", "A.", "A,", "(A)"]
        if answer == "A" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("A")
    if "B" in valid_option:
        characters = ["A", "C", "D", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["B)", "B.", "B,", "(B)"]
        if answer == "B" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("B")
    if "C" in valid_option:
        characters = ["A", "B", "D", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["C)", "C.", "C,", "(C)"]
        if answer == "C" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("C")
    if "D" in valid_option:
        characters = ["A", "B", "C", "E", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["D)", "D.", "D,", "(D)"]
        if answer == "D" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("D")
    if "E" in valid_option:
        characters = ["A", "B", "C", "D", "F", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["E)", "E.", "E,", "(E)"]
        if answer == "E" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("E")
    if "F" in valid_option:
        characters = ["A", "B", "C", "D", "E", "G"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["F)", "F.", "F,", "(F)"]
        if answer == "F" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("F")
    if "G" in valid_option:
        characters = ["A", "B", "C", "D", "E", "F"]
        combinations = [char + punct for char in characters for punct in punctuations]
        start_patterns = ["G)", "G.", "G,", "(G)"]
        if answer == "G" or (any(answer.startswith(pattern) for pattern in start_patterns) and all(x not in answer for x in combinations)):
            ch_cand_list.append("G")
    if len(ch_cand_list) == 1:
        return ch_cand_list[0]

    return False


# Function based on 'https://github.com/AtsuMiyai/UPD' code
def build_option_str_w_gt(option_list, gt_text, eval_type_specific, question_type, upd_type):
    chars = string.ascii_uppercase
    s = 'There are several options: \n'
    valid_option = []
    answer_option = []

    for c, opt in zip(chars, option_list):
        if is_none(opt) is False:
            s += f'{c}. {opt}\n'
            valid_option.append(c)
    if upd_type == "aad":
        if eval_type_specific == "aad":
            gt_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    gt_option = option
                    break
            none_option_mapping = {'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
            s += f'{gt_option}. {gt_text}\n'
            none_option = none_option_mapping[gt_option]
            s += f'{none_option}. The correct answer is No answer, None of the above, \
                all provided options are incorrect, or I cannot answer.\n'
            valid_option.append(gt_option)
            valid_option.append(none_option)
            answer_option = [gt_option, none_option]
            if question_type == 'inst':
                s += f'F. None of the above\n'
                valid_option.append("F")
                answer_option = [gt_option, none_option, "F"]

        if eval_type_specific == "standard":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is No answer, None of the above, \
                all provided options are incorrect, or I cannot answer.\n'
            valid_option.append(none_option)
            if question_type == 'inst':
                s += f'F. None of the above\n'
                valid_option.append("F")
    elif upd_type == "iasd":
        if eval_type_specific == "iasd":
            gt_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    gt_option = option
                    break

            s += f'{gt_option}. {gt_text}\n'
            valid_option.append(gt_option)

            if question_type == 'inst':
                if gt_option == 'E':
                    s += f'F. None of the above\n'
                    valid_option.append('F')
                    s += 'G. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                    valid_option.append('G')
                    answer_option = [gt_option, 'F', 'G']
                else:
                    none_option_mapping = {'C': 'D', 'D': 'E'}
                    none_option = none_option_mapping[gt_option]
                    s += f'{none_option}. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                    valid_option.append(none_option)
                    s += f'F. None of the above\n'
                    valid_option.append('F')
                    answer_option = [gt_option, none_option, 'F']
            else:
                none_option_mapping = {'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G'}
                none_option = none_option_mapping[gt_option]
                s += f'{none_option}. The correct answer is No answer, None of the above, all provided options are irrelevant or incorrect, or I cannot answer.\n'
                valid_option.append(none_option)
                answer_option = [gt_option, none_option]

        if eval_type_specific == "standard":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is No answer, None of the above, \
                all provided options are irrelevant or incorrect, or I cannot answer.\n'
            valid_option.append(none_option)
            if question_type == 'inst':
                s += f'F. None of the above\n'
                valid_option.append("F")
    elif upd_type == "ivqd":
        if eval_type_specific == "ivqd":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is that The image is incompatible with the question, or I cannot answer.\n'
            valid_option.append(none_option)
            answer_option = [none_option]
            if question_type == 'inst':
                s += f'F. The image and question are irrelevant.\n'
                valid_option.append("F")
                answer_option = [none_option, "F"]

        if eval_type_specific == "standard":
            none_option = ""
            for option in ['C', 'D', 'E', 'F']:
                if option not in valid_option:
                    none_option = option
                    break
            s += f'{none_option}. The correct answer is that The image is incompatible with the question, or I cannot answer.\n'
            valid_option.append(none_option)
            if question_type == 'inst':
                s += f'F. The image and question are irrelevant.\n'
                valid_option.append("F")
    else:
        assert False

    return s, valid_option, answer_option


# Function based on 'https://github.com/AtsuMiyai/UPD' code
def build_prompt_for_gpt(question, options, prediction, upd_type):
    if upd_type == "aad":
        tmpl = (
            'You are an AI assistant who will help me to match an answer '
            'with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different '
            'from the answer, output H. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G '
            '(if they are valid options), and H. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. no answer \nAnswer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above  \nAnswer: no answer \nYour output: E\n'
            'Example 3: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above \nAnswer: fish \nYour output: H\n'
            'Example 4: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
    elif upd_type == "iasd":
        tmpl = (
            'You are an AI assistant who will help me to match an answer '
            'with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different '
            'from the answer, output H. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G '
            '(if they are valid options), and H. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. no answer \nAnswer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above  \nAnswer: no answer \nYour output: E\n'
            'Example 3: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. None of the above \nAnswer: fish \nYour output: H\n'
            'Example 4: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
    elif upd_type == "ivqd":
        tmpl = (
            'You are an AI assistant who will help me to match an answer '
            'with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different '
            'from the answer, output H. '
            'Your should output a single uppercase character in A, B, C, D, E, F, G '
            '(if they are valid options), and H. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: The updloaded image and question are incompatible. \nYour output: E\n'
            'Example 3: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear '
            'B. rabbit C. cat D. dog E. The image and question are irrelevant \nAnswer: fish \nYour output: H\n'
            'Example 4: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: ')
    else:
        assert False

    return tmpl.format(question, options, prediction)


def get_multiple_choice_answer_from_gpt(open_ai_client, prediction, question, options, choices, gt_text, upd_setting, eval_type_specific, upd_type):
    # 1. First, try to prefetch from string matching again, with considering valid_option
    option_str, valid_option, correct_answer_options = build_option_str_w_gt(option_list=options, gt_text=gt_text,
                                                                    eval_type_specific=eval_type_specific,
                                                                    question_type=upd_setting, upd_type=upd_type)

    prefetched_answer = infer_option_by_string_matching(prediction, choices, valid_option=valid_option)
    if prefetched_answer:
        return prefetched_answer, correct_answer_options, False

    # 2. If the above does not work, try to GPT
    prompt_to_gpt = build_prompt_for_gpt(question=question, options=option_str, prediction=prediction, upd_type=upd_type)

    retry = 3
    while retry:
        answer_from_gpt = call_open_ai_gpt_3_5(open_ai_client, prompt_to_gpt)
        prefetched_answer = infer_option_by_string_matching(answer_from_gpt, choices, valid_option=valid_option)
        if prefetched_answer:
            break
        retry -= 1

    return prefetched_answer, correct_answer_options, True


def is_correct_after_gpt(prefetched_answer, correct_answer_options, gt_answer, eval_type_specific):
    if eval_type_specific == "standard":
        is_correct = int(prefetched_answer == gt_answer)
    elif eval_type_specific in ["aad", "iasd", "ivqd"]:
        is_correct = int(prefetched_answer in correct_answer_options)
    else:
        assert False

    return is_correct


def evaluate_single_multiple_choice_prediction(question, prediction, gt_answer, gt_text, eval_type_specific, choices, options, open_ai_client, upd_setting, upd_type):
    did_call_to_get = False

    # 1. If prediction is an empty string, the answer is incorrect
    if len(prediction) == 0:
        is_correct = 0
        prefetched_answer = prediction
        return prefetched_answer, is_correct, did_call_to_get

    # 2. Try to infer answer by string template
    prefetched_answer = infer_option_by_string_matching(answer=prediction, option_dict=choices, question_type=upd_setting)
    if prefetched_answer:
        is_correct = int(prefetched_answer == gt_answer)
        return prefetched_answer, is_correct, did_call_to_get

    # 3. If this is a standard answerable question, and the prediction is "I cannot answer.", the prediction is incorrect
    if eval_type_specific == 'standard' and prediction == i_cannot_answer_str:
        is_correct = 0
        prefetched_answer = prediction
        return prefetched_answer, is_correct, did_call_to_get

    # 4. If this is an unanswerable question, and the prediction is "I cannot answer.", the prediction is correct
    if eval_type_specific in ['ivqd', 'aad', 'iasd'] and prediction == i_cannot_answer_str:
        is_correct = 1
        prefetched_answer = gt_answer
        return prefetched_answer, is_correct, did_call_to_get

    # 5. If all the above does not succeed, use GPT
    print('Getting answer from GPT...')

    # Get answer from GPT
    prefetched_answer, correct_answer_options, did_call_to_get = (
        get_multiple_choice_answer_from_gpt(open_ai_client, prediction, question, options, choices=choices,
                                            gt_text=gt_text, upd_setting=upd_setting,
                                            eval_type_specific=eval_type_specific, upd_type=upd_type))
    # Find if the answer is correct
    is_correct = is_correct_after_gpt(prefetched_answer=prefetched_answer,
                                      correct_answer_options=correct_answer_options, gt_answer=gt_answer,
                                      eval_type_specific=eval_type_specific)

    return prefetched_answer, is_correct, did_call_to_get


def evaluate_multiple_choice_results(open_ai_client, standard_results, upd_results, upd_type, upd_setting, epoch, results_eval_info_file_path=None):
    results_eval_info_file = open(results_eval_info_file_path, "w") if results_eval_info_file_path is not None else None

    # 1. Evaluate standards
    standard_accuracy, standard_hits, standard_total, circ_standard_accuracy, circ_standard_hits, \
        circ_standard_total, circ_standard_hits_dict, standard_calls_to_gpt = evaluate_standard_or_upd_multiple_choice_results(
        results_list=standard_results, open_ai_client=open_ai_client, standard_or_upd='standard', upd_type=upd_type,
        upd_setting=upd_setting, epoch=epoch, results_eval_info_file=results_eval_info_file)

    # 2. Evaluate UPDs
    upd_accuracy, upd_hits, upd_total, circ_upd_accuracy, circ_upd_hits, \
        circ_upd_total, circ_upd_hits_dict, upd_calls_to_gpt = evaluate_standard_or_upd_multiple_choice_results(
        results_list=upd_results, open_ai_client=open_ai_client, standard_or_upd='upd', upd_type=upd_type,
        upd_setting=upd_setting, epoch=epoch, results_eval_info_file=results_eval_info_file)

    if results_eval_info_file is not None:
        results_eval_info_file.close()

    # 3. Evaluate dual and print
    dual_accuracy, dual_hits, dual_total, dual_hits_dict = evaluate_dual_accuracy(circ_standard_hits_dict,
                                                                                      circ_upd_hits_dict)

    print_dual_accuracy_scores(dual_accuracy=dual_accuracy, dual_total=dual_total, dual_hits=dual_hits, dual_hits_dict=dual_hits_dict,
                               upd_type_str_for_print=upd_type.upper(), epoch=epoch)

    num_gpt_calls_to_gpt = standard_calls_to_gpt + upd_calls_to_gpt

    return standard_accuracy, circ_standard_accuracy, upd_accuracy, circ_upd_accuracy, dual_accuracy, num_gpt_calls_to_gpt


def evaluate_standard_or_upd_multiple_choice_results(results_list, open_ai_client, standard_or_upd, upd_type,
                                                     upd_setting, epoch, results_eval_info_file):
    # 1. Evaluate all results
    regular_hits, regular_total = 0, 0
    circular_items_dict = defaultdict(list)
    calls_to_gpt = 0

    for res_dict in results_list:
        prediction = res_dict['output']
        choices = res_dict['choices']
        gt_answer = res_dict['original_gt_answer']
        question = res_dict['question_str']
        options = res_dict['options']
        eval_type = res_dict['type']
        question_id = res_dict['question_id']

        eval_type_specific = get_eval_type_specific(eval_type=eval_type, upd_type=upd_type)
        gt_text = res_dict['masked_answer'] if eval_type_specific in ["aad", "iasd"] else None

        prefetched_answer, is_correct, did_call_to_gpt = evaluate_single_multiple_choice_prediction(question=question,
                                                                                                    prediction=prediction,
                                                                                                    gt_answer=gt_answer,
                                                                                                    eval_type_specific=eval_type_specific,
                                                                                                    gt_text=gt_text,
                                                                                                    choices=choices, options=options,
                                                                                                    open_ai_client=open_ai_client,
                                                                                                    upd_setting=upd_setting,
                                                                                                    upd_type=upd_type)
        print(f"{upd_type.upper()} result: {standard_or_upd} output: question_id = {question_id}, prediction = \"{prediction}\", gt_answer = \"{gt_answer}\", is_correct: {is_correct}")

        res_dict['prefetched_answer'] = prefetched_answer
        res_dict['is_correct'] = is_correct

        circular_item_index = get_circular_item_index(question_id)
        circular_items_dict[circular_item_index].append(res_dict)

        regular_total += 1
        calls_to_gpt += int(did_call_to_gpt)
        regular_hits += int(is_correct)

        if results_eval_info_file is not None:
            results_eval_info_file.write(json.dumps(res_dict) + "\n")
            results_eval_info_file.flush()

    # 2. Compute regular accuracy
    regular_accuracy = regular_hits / regular_total

    # 3. Compute circular accuracy and other stats
    circular_accuracy, circular_hits, circular_total, circular_items_hits = evaluate_circular_accuracy(
        circular_items_dict)

    # 4. Print
    print_regular_and_circular_accuracy_scores(epoch, regulars_total=regular_total,
                                               regulars_accuracy=regular_accuracy, regulars_hits=regular_hits,
                                               circulars_total=circular_total,
                                               circulars_accuracy=circular_accuracy,
                                               circulars_hits=circular_hits,
                                               circulars_hits_dict=circular_items_hits,
                                               upd_type_str_for_print=upd_type.upper(),
                                               standard_or_upd=standard_or_upd)

    return regular_accuracy, regular_hits, regular_total, circular_accuracy, circular_hits, circular_total, circular_items_hits, calls_to_gpt
