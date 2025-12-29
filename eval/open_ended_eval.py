import json

from prompting.prompting_utils import i_cannot_answer_str
from eval.eval_utils import print_dual_accuracy_scores, get_eval_type_specific, \
    print_regular_and_circular_accuracy_scores, evaluate_dual_accuracy
from eval.lave_metric.lave_gpt_3_5 import LaveGPT_3_5


def evaluate_single_open_ended_prediction(prediction, gt_answer, question, lave_metric, eval_type_specific):
    prediction_formatted = prediction.rstrip('.').lower()
    gt_answer_formatted = gt_answer.rstrip('.').lower()

    did_call_to_lave = False
    answer_from_lave = None
    is_perfect_match = (prediction_formatted == gt_answer_formatted)
    if is_perfect_match:
        score = 1
    else:
        if eval_type_specific == 'standard' and prediction == i_cannot_answer_str:
            score = 0
        else:
            print('Getting answer from LAVE metric...')
            did_call_to_lave = True
            score, answer_from_lave = lave_metric.compute(
                prediction=prediction,
                references=[gt_answer],
                question=question
            )

    return score, did_call_to_lave, answer_from_lave


def evaluate_open_ended_standard_or_upd_results(results, data_type_str, upd_type, lave_metric, epoch, results_eval_info_file):
    regulars_hits, regulars_total = 0, 0
    items_hits_dict = {}
    num_calls_to_lave = 0
    for res in results:
        prediction = res['output']
        gt_answer = res['original_gt_answer']
        question = res['question_str']

        eval_type = res['type']
        eval_type_specific = get_eval_type_specific(eval_type=eval_type, upd_type=upd_type)

        is_correct_score, did_call_to_lave, answer_from_lave = \
            evaluate_single_open_ended_prediction(prediction=prediction, gt_answer=gt_answer, question=question,
                                                  lave_metric=lave_metric, eval_type_specific=eval_type_specific)
        res['is_correct'] = is_correct_score

        print(f"{data_type_str} output: question_id = {res['question_id']}, question: \"{question}\", prediction: \"{prediction}\", gt_answer: \"{gt_answer}\", is_correct: {is_correct_score}")

        regulars_hits += is_correct_score
        regulars_total += 1

        if results_eval_info_file is not None:
            results_eval_info_file.write(json.dumps(res) + "\n")
            results_eval_info_file.flush()

        num_calls_to_lave += int(did_call_to_lave)
        items_hits_dict[res['question_id']] = is_correct_score

    regulars_accuracy = regulars_hits / regulars_total

    print_regular_and_circular_accuracy_scores(epoch=epoch, regulars_total=regulars_total,
                                               regulars_accuracy=regulars_accuracy, regulars_hits=regulars_hits,
                                               circulars_total=0, circulars_accuracy=0, circulars_hits=0,
                                               circulars_hits_dict={},
                                               upd_type_str_for_print=upd_type.upper(),
                                               standard_or_upd=data_type_str, with_circular=False)

    return items_hits_dict, num_calls_to_lave, regulars_total, regulars_hits, regulars_accuracy


def evaluate_open_ended_results(open_ai_client, standard_results, upd_results, upd_type, epoch, results_eval_info_file_path=None):
    lave_metric = LaveGPT_3_5(open_ai_client=open_ai_client)
    results_eval_info_file = open(results_eval_info_file_path, "w") if results_eval_info_file_path is not None else None

    # 1. Evaluate standards
    standards_hits_dict, standard_calls_to_lave, standard_total, standard_prefect_hits, standard_accuracy = \
        evaluate_open_ended_standard_or_upd_results(results=standard_results, data_type_str='standard',
                                                    upd_type=upd_type, lave_metric=lave_metric, epoch=epoch,
                                                    results_eval_info_file=results_eval_info_file)

    # 2. Evaluate UPDs
    upds_hits_dict, upd_calls_to_lave, upd_total, upd_prefect_hits, upd_accuracy = \
        evaluate_open_ended_standard_or_upd_results(results=upd_results, data_type_str='upd',
                                                    upd_type=upd_type, lave_metric=lave_metric, epoch=epoch,
                                                    results_eval_info_file=results_eval_info_file)

    if results_eval_info_file is not None:
        results_eval_info_file.close()

    # 3. Evaluate dual and print
    dual_accuracy, dual_hits, dual_total, dual_hits_dict = evaluate_dual_accuracy(standards_hits_dict,
                                                                                  upds_hits_dict)
    print_dual_accuracy_scores(dual_accuracy=dual_accuracy, dual_total=dual_total, dual_hits=dual_hits,
                               dual_hits_dict=dual_hits_dict,
                               upd_type_str_for_print=upd_type.upper(), epoch=epoch)

    num_calls_to_lave = standard_calls_to_lave + upd_calls_to_lave

    return standard_accuracy, upd_accuracy, dual_accuracy, num_calls_to_lave
