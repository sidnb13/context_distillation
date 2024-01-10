import os
import pickle as pkl

if __name__ == "__main__":
    tasks = [
             ('task1631_openpi_answer_generation', True), 
             ('task1631_openpi_answer_generation', False), 
             ('task039_qasc_find_overlapping_words', True), 
             ('task039_qasc_find_overlapping_words', False), 
             ('task1157_bard_analogical_reasoning_rooms_for_containers', True), 
             ('task1157_bard_analogical_reasoning_rooms_for_containers', False), 
             ('task1158_bard_analogical_reasoning_manipulating_items', True), 
             ('task1158_bard_analogical_reasoning_manipulating_items', False), 
             ('task1516_imppres_naturallanguageinference', True), 
             ('task1516_imppres_naturallanguageinference', False), 
             
             ('task034_winogrande_question_modification_object', True), 
             ('task034_winogrande_question_modification_object', False), 
             ('task1540_parsed_pdfs_summarization', True), 
             ('task1540_parsed_pdfs_summarization', False), 
             ('task418_persent_title_generation', True), 
             ('task418_persent_title_generation', False), 
             ('task401_numeric_fused_head_reference', True), 
             ('task401_numeric_fused_head_reference', False), 
             ('task891_gap_coreference_resolution', True), 
             ('task891_gap_coreference_resolution', False), 
            ]
    
    for task, add_explanation in tasks:
        print('='*25)
        print(task, add_explanation)
        print('='*25)

        data_out_path = "../../outputs/tk_inject_2_negative_%s___%s/" % ('plus_explanation' if add_explanation else '', task)

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
        
