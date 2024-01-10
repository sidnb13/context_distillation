import os
import pickle as pkl

if __name__ == "__main__":

    tasks = [
        ('task1624_disfl_qa_question_yesno_classification', True, True), 
        ('task1624_disfl_qa_question_yesno_classification', False, True), 
        ('task1624_disfl_qa_question_yesno_classification', False, False), 
        ('task970_sherliic_causal_relationship', True, True), 
        ('task970_sherliic_causal_relationship', False, True), 
        ('task970_sherliic_causal_relationship', False, False), 
        ('task1516_imppres_naturallanguageinference', True, True), 
        ('task1516_imppres_naturallanguageinference', False, True), 
        ('task1516_imppres_naturallanguageinference', False, False), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', True, True), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', False, True), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', False, False), 
        ('task362_spolin_yesand_prompt_response_sub_classification', True, True), 
        ('task362_spolin_yesand_prompt_response_sub_classification', False, True), 
        ('task362_spolin_yesand_prompt_response_sub_classification', False, False), 

        ('task1586_scifact_title_generation', True, True), 
        ('task1586_scifact_title_generation', False, True), 
        ('task1586_scifact_title_generation', False, False), 
        ('task743_eurlex_summarization', True, True), 
        ('task743_eurlex_summarization', False, True), 
        ('task743_eurlex_summarization', False, False), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', True, True), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', False, True), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', False, False), 
        ('task033_winogrande_answer_generation', True, True), 
        ('task033_winogrande_answer_generation', False, True), 
        ('task033_winogrande_answer_generation', False, False), 
        ('task937_defeasible_nli_social_classification', True, True), 
        ('task937_defeasible_nli_social_classification', False, True), 
        ('task937_defeasible_nli_social_classification', False, False), 
    ]
    
    for task, add_explanation, add_negatives in tasks:
        print('RUNNING:')
        print('='*25)
        print(task, add_explanation, add_negatives)
        print('='*25)

        data_out_path = "../../outputs/inject2negv2negexpl/tk_inject_%s_%s___%s/" % ('2_negative' if add_negatives else '', 'plus_explanation' if add_explanation else '', task)

        with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'rb') as f:
            teacher_acc = pkl.load(f)[task]
        with open(os.path.join(data_out_path, 'pre_distill_student_accuracies.pkl'), 'rb') as f:
            pre_distill_student_acc = pkl.load(f)[task]
        with open(os.path.join(data_out_path, 'post_distill_student_accuracies.pkl'), 'rb') as f:
            post_distill_student_acc = pkl.load(f)[task]
        
        print('summary:')
        print('teacher accuracy:', teacher_acc)
        print('pre-distill student accuracy:', pre_distill_student_acc)
        print('post-distill student accuracy:', post_distill_student_acc)
        print('='*25)