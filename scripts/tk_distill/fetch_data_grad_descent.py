import pickle as pkl
import os

if __name__ == "__main__":

    tasks = [
        ('task1391_winogrande_easy_answer_generation', 1), 
        ('task1391_winogrande_easy_answer_generation', 2), 
        ('task1391_winogrande_easy_answer_generation', 4), 
        ('task1391_winogrande_easy_answer_generation', 8), 
        ('task1391_winogrande_easy_answer_generation', 16), 
        ('task1391_winogrande_easy_answer_generation', 32), 
    ]

    max_epochs = 25
    
    for task, n_training_data in tasks:
        print('RUNNING:')
        print('='*25)
        print(task, n_training_data)
        print('='*25)

        data_out_path = "../../outputs/injectionpromptvgradientdescent/tk_gradient_descent_%d_datapoints____%s/" % (n_training_data, task)

        datas = []
        for x in range(max_epochs):
            with open(os.path.join(data_out_path, 'post_distill_model_accuracies_epoch_%d.pkl' % (x)), 'rb') as f:
                datas.append(pkl.load(f)[task])
        
        with open(os.path.join(data_out_path, 'all_teacher_accuracies.pkl'), 'rb') as f:
            teacher_datas = list(map(lambda x: x[task], pkl.load(f)))
        
        max_rougeL = max(map(lambda x: x['rougeL'], datas))
        max_exact_match = max(map(lambda x: x['exact_match'], datas))
        max_rouge1 = max(map(lambda x: x['rouge1'], datas))

        print('max_rougeL', max_rougeL)
        print('max_exact_match', max_exact_match)
        print('max_rouge1', max_rouge1)

    print('teacher:')
    print('max_rougeL', max_rougeL)
    print('max_exact_match', max_exact_match)
    print('max_rouge1', max_rouge1)


