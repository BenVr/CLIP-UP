from ratelimit import limits, sleep_and_retry


def get_eval_type_specific(eval_type, upd_type):
    if eval_type == 'standard':
        return eval_type
    elif eval_type == 'upd':
        return upd_type
    else:
        assert False


def evaluate_circular_accuracy(circular_items_dict):
    circular_hits, circular_total = 0, 0
    circular_items_hits = {}
    for circular_item_idx, circular_items_list in circular_items_dict.items():
        are_all_circular_items_correct = int(all(res['is_correct'] == 1 for res in circular_items_list))
        circular_hits += are_all_circular_items_correct
        circular_total += 1
        circular_items_hits[circular_item_idx] = are_all_circular_items_correct

    circular_accuracy = circular_hits / circular_total
    circular_items_hits = dict(sorted(circular_items_hits.items()))

    return circular_accuracy, circular_hits, circular_total, circular_items_hits


def evaluate_dual_accuracy(circular_standard_hits, circular_upd_hits):
    assert circular_standard_hits.keys() == circular_upd_hits.keys()

    dual_hits, dual_total = 0, 0
    dual_hits_dict = {}
    for circular_item_index in circular_standard_hits.keys():
        # For multiple-choice VQA, the below row is like
        # "is_dual_accurate == int(circular_standard_hits[circular_item_index] == 1 and circular_upd_hits[circular_item_index] == 1)";
        # For open-ended VQA, it is not the same, since LAVE evaluation may score 0.5
        is_dual_correct = min(circular_standard_hits[circular_item_index], circular_upd_hits[circular_item_index])

        dual_hits += is_dual_correct
        dual_total += 1
        dual_hits_dict[circular_item_index] = is_dual_correct

    dual_accuracy = dual_hits / dual_total

    return dual_accuracy, dual_hits, dual_total, dual_hits_dict


# Set limit: 500 requests per minute to not issue too many calls
@sleep_and_retry
@limits(calls=500, period=60)
def call_open_ai_gpt_3_5(open_ai_client, input_str):
    try:
        chat_completion = open_ai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_str,
                }
            ],
            model="gpt-3.5-turbo-0125",
        )
        predicted_answer = chat_completion.choices[0].message.content
    except Exception as err:
        print(f"An error with calling OpenAI occurred: {err}")
        predicted_answer = ''

    return predicted_answer


def print_regular_and_circular_accuracy_scores(epoch, regulars_total, regulars_accuracy, regulars_hits,
                                               circulars_total, circulars_accuracy, circulars_hits,
                                               circulars_hits_dict, upd_type_str_for_print, standard_or_upd,
                                               with_circular=True):

    epoch_str = f'Epoch {epoch}, ' if epoch is not None else ''
    print(f'{epoch_str}{upd_type_str_for_print} evaluation: Evaluating {regulars_total} {standard_or_upd} samples '
          f'(all items; no consideration of circular items): '
          f'Acc: {regulars_accuracy * 100:.2f}% (solved {regulars_hits} samples).')

    if with_circular:
        print(f'{epoch_str}{upd_type_str_for_print} evaluation: Evaluating {circulars_total} {standard_or_upd} circular '
              f'samples: Acc: {circulars_accuracy * 100:.2f}% (solved {circulars_hits} samples).')

    return


def print_dual_accuracy_scores(dual_accuracy, dual_total, dual_hits, dual_hits_dict, upd_type_str_for_print, epoch=None):
    epoch_str = f'Epoch {epoch}, ' if epoch is not None else ''

    print(f'{epoch_str}{upd_type_str_for_print} evaluation: Evaluating {dual_total} samples for dual accuracy: Acc: {dual_accuracy * 100:.2f}% '
          f'(solved {dual_hits} samples).')

    return

