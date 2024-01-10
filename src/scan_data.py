from datasets import load_dataset
from utils.randomness import RandomState, seed_context
from injection_functions import InjectionData, format_input
import random

def load_scan():
    # Downloading the datasets
    # also included the length split since it contains super short commands so that we can setup the incremental learning
    train_length_scan_dataset = load_dataset("scan", "length")['train']
    test_length_scan_dataset = load_dataset("scan", "length")['train']
    simple_test = load_dataset("scan", "simple")['test']

    simple_test = list(simple_test)
    train_dataset = list(train_length_scan_dataset) + list(test_length_scan_dataset)
    all_train_x = {d['commands'] for d in train_dataset}
    new_test_set = [l for l in simple_test if l['commands'] not in all_train_x]

    all_vocab = set()
    for c in all_train_x:
        all_vocab |= set(c.split(' '))
    all_vocab = list(all_vocab)
    print(all_vocab)

    curriculum = ['run', 'jump', 'and', 'walk', 'look', 'left', 'right', 'turn', 'after', 'opposite', 'twice', 'thrice', 'around']
    print(set(curriculum) == set(all_vocab))
    def get_curriculum(new_l=curriculum):
        datapoint_by_new_vocab = []
        for i in range(len(new_l)):
            existing_vocab = set(new_l[:i + 1])
            cur_vocab = new_l[i]
            demo_example = []
            for l in train_dataset:
                c = l['commands']
                if cur_vocab in c and all(v in existing_vocab for v in c.split(' ')):
                    demo_example.append(l)
            if len(demo_example) == 0:
                print('empty')
                print(cur_vocab)
                print(existing_vocab)
            datapoint_by_new_vocab.append(demo_example)
        return datapoint_by_new_vocab

    datapoints_by_curriculum = get_curriculum(curriculum)
    print([len(x) for x in datapoints_by_curriculum])

    explanations = {
        'run': 'run will be translated to I_RUN',
        'jump': 'jump will be translated to I_JUMP',
        'and': 'A and B will be translated to "translation of A" + "translation of B". For example, jump and run will be translated to I_JUMP I_RUN',
        'walk': 'walk will be translated to I_WALK',
        'look': 'look will be translated to I_LOOK',
        'left': 'when we use left after a verb, we first turn left and perform the action. For example, jump left will translate to I_TURN_LEFT I_JUMP',
        'right': 'when we use right after a verb, we first turn right and perform the action. For example, jump right will translate to I_TURN_RIGHT I_JUMP',
        'turn': 'turn is an action, which will be followed by left or right. turn left corresponds to I_TURN_LEFT, and turn right corresponds to I_TURN_RIGHT',
        'after': 'A after B will be translated to "translation of B" + "translation of A". For example, jump after run will be translated to I_RUN I_JUMP',
        'opposite': 'opposite is used before right and left, and corresponds to turning left/right twice, before committing an action. For example, turn opposite left will be translated to I_TURN_LEFT I_TURN_LEFT, and look opposite right will be translated to I_TURN_RIGHT I_TURN_RIGHT I_LOOK',
        'twice': 'twice meanings repeating an action twice. For example, jump twice will be translated into I_JUMP I_JUMP',
        'thrice': 'thrice meanings repeating an action three times. For example, run thrice will be translated into I_RUN I_RUN I_RUN',
        'around': 'around is usually used after a verb and before a direction; this means performing an action and turn into a particular direction, and do them four times in total. For example, run around left should be translated to I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN'
    }
    # assert set()

    return curriculum, datapoints_by_curriculum, new_test_set, explanations

def create_scan_cirriculum(add_explanation: bool, n_positive: int, seed: int):
    random_state = RandomState(seed)
    curriculum, datapoints_by_curriculum, test_set, explanations = load_scan()

    all_datas = []

    all_train_data = sum(map(lambda x: list(zip(x[1], [curriculum[x[0]]]*len(x[1]))), enumerate(datapoints_by_curriculum)), [])
    for datapoints in datapoints_by_curriculum:
        with seed_context(random_state):
            random.shuffle(datapoints)
    
    with seed_context(random_state):
        random.shuffle(test_set)
        random.shuffle(all_train_data)

    for z, (new_vocab, datapoints) in enumerate(zip(curriculum, datapoints_by_curriculum)):

        injection_datas = datapoints[:n_positive]
        teacher_prompt = f"Definition: Given a sequence of actions to navigate an agent in its environment, provide the correct command in a limited form of natural language that matches the sequence of actions when executed. Commands are lowercase and encapsulate the logic of the sequence of actions. Actions are individual steps that serve as the building blocks for a command. There are only six actions: 'I_LOOK', 'I_WALK', 'I_RUN', 'I_JUMP', 'I_TURN_LEFT', and 'I_TURN_RIGHT'. These actions respectively align with the commands 'look', 'walk', 'run', 'jump', 'turn left', and 'turn right'. For commands, 'left' and 'right' are used to denote the direction of an action. opposite turns the agent backward in the specified direction. The word 'around' makes the agent execute an action while turning around in the specified direction. The word 'and' means to execute the next scope of the command following the previous scope of the command. The word 'after' signifies to execute the previous scope of the command following the next scope of the command. The words 'twice' and 'thrice' trigger repetition of a command that they scope over two times or three times, respectively. Actions and commands do not have quotations in the input and output."
        for i, injection_data in enumerate(injection_datas):
            teacher_prompt += f" Positive Example {i+1} – Input: {injection_data['commands']} . Output: {injection_data['actions']} ."
            if add_explanation:
                teacher_prompt += f" Explanation: {explanations[new_vocab]}."
        
        # questions = list(map(lambda x: x['commands'], all_train_data_no_prompt))
        questions = sum([list(map(lambda x: x['commands'], datapoints_by_curriculum[x])) for x in range(z+1)], [])
        
        all_datas.append(InjectionData(
            teacher_prompt=teacher_prompt, 
            student_prompt="", 
            question_generation_prompts=[], 
            # ask random questions not in the prompt (maybe change this)
            dataset_questions=questions, 
            dataset_eval=[(format_input(example['commands']), [example['actions']]) for example in datapoints[n_positive:]][:128], 
            meta=None, 
        ))
    
    
    injection_datas = random.sample(all_train_data, n_positive)
    teacher_prompt = f"Definition: Given a sequence of actions to navigate an agent in its environment, provide the correct command in a limited form of natural language that matches the sequence of actions when executed. Commands are lowercase and encapsulate the logic of the sequence of actions. Actions are individual steps that serve as the building blocks for a command. There are only six actions: 'I_LOOK', 'I_WALK', 'I_RUN', 'I_JUMP', 'I_TURN_LEFT', and 'I_TURN_RIGHT'. These actions respectively align with the commands 'look', 'walk', 'run', 'jump', 'turn left', and 'turn right'. For commands, 'left' and 'right' are used to denote the direction of an action. opposite turns the agent backward in the specified direction. The word 'around' makes the agent execute an action while turning around in the specified direction. The word 'and' means to execute the next scope of the command following the previous scope of the command. The word 'after' signifies to execute the previous scope of the command following the next scope of the command. The words 'twice' and 'thrice' trigger repetition of a command that they scope over two times or three times, respectively. Actions and commands do not have quotations in the input and output."
    for i, (injection_data, new_vocab) in enumerate(injection_datas):
        teacher_prompt += f" Positive Example {i+1} – Input: {injection_data['commands']} . Output: {injection_data['actions']} ."
        if add_explanation:
            teacher_prompt += f" Explanation: {explanations[new_vocab]}."
    
    eval_data = InjectionData(
        teacher_prompt=teacher_prompt, 
        student_prompt="", 
        question_generation_prompts=sorted(list(set(map(lambda x: x[0]['commands'], all_train_data)).difference(set(map(lambda x: x[0]['commands'], injection_datas))))), 
        dataset_questions=list(map(lambda x: x[0]['commands'], all_train_data)), 
        dataset_eval=[(format_input(example['commands']), [example['actions']]) for example in test_set[:128]], 
        meta=None, 
    )

    return all_datas, eval_data

if __name__ == "__main__":
    all_datas, eval_data = create_scan_cirriculum(add_explanation=True, n_positive=2, seed=42)
