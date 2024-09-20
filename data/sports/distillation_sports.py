from typing import List, Optional
import fire
import json
from tqdm import tqdm
from llama import Llama, Dialog


def generate_(
    data,
    generator,
    max_batch_size,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None):

    for i in tqdm(range(int(len(data)/max_batch_size)+1), mininterval=2, desc='  - (Generating)   ', leave=False):
        input_list = []
        for sample in data[i*max_batch_size:(i+1)*max_batch_size]:
            input_list.append([
                {"role": "system", "content": "Use two extremely short sentences to reply. "
                                          "The first one is '1. The user prefers xxx .'"
                                          "The second one is '2. The item's attributes are xxx."},
                {
                    "role": "user",
                    "content": "A user bought an item and said \"{explanation}\".  "
                               "Use two sentences to explain the user's preference and the item's attributions, respectively. ".format(
                        explanation=sample['explanation']),
                },
            ])

        dialogs: List[Dialog] = input_list
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for j, (dialog, result) in enumerate(zip(dialogs, results)):
            strs = result['generation']['content'].split('\n')
            try:
                # if len(strs) > 1 and strs[0].startswith(' The user'):
                #     print(strs)
                data[i * max_batch_size + j]['user_preference'] = strs[1][3:]
                data[i * max_batch_size + j]['item_attribution'] = strs[2][3:]
            except Exception as e:

                try:
                    if strs[1].contains('1. '):

                        # for the case that
                        # 1. The user prefers a hair cutting tool that can handle long hair,
                        # and the item's attributes include a long cutting length for ease of use.
                        strss = strs[1].split(', and ')
                        if len(strss) > 1:
                            data[i * max_batch_size + j]['user_preference'] = strss[0][3:]
                            data[i * max_batch_size + j]['item_attribution'] = strss[1]
                            continue

                        # for the case like:
                        # 1. The user prefers item 2 because it is better than a dozen pins.
                        # (2. The item's attributes are better than a dozen pins.)
                        strss = strs[1].index('(2. ')
                        if len(strss) > 1:
                            data[i * max_batch_size + j]['user_preference'] = strss[0][3:]
                            data[i * max_batch_size + j]['item_attribution'] = strss[1]
                            continue

                    print(result['generation']['content'])
                    data[i * max_batch_size + j]['user_preference'] = data[i * max_batch_size + j]['explanation']
                    data[i * max_batch_size + j]['item_attribution'] = data[i * max_batch_size + j]['explanation']

                except Exception as e:
                    data[i * max_batch_size + j]['user_preference'] = data[i * max_batch_size + j]['explanation']
                    data[i * max_batch_size + j]['item_attribution'] = data[i * max_batch_size + j]['explanation']

            # # for msg in dialog:
            # #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            # # print(
            # #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            # # )
            # #
            # # print('-=-=-=-=')
            # # print(' '.join(result['generation']['content'].split('\n')[1:]))

            # print("\n==================================\n")
    return data

def main(
    ckpt_dir: str = 'llama-2-7b-chat/',
    tokenizer_path: str = 'tokenizer.model',
    max_seq_len: int = 512,
    max_batch_size: int = 200):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    f = open('./data/sports/explanation.json', 'r')
    data = json.load(f)
    f.close()

    #

    training_data = data['train']
    val_data = data['val']
    test_data = data['test']

    new_data = {
        'train': generate_(
            training_data,
            generator,
            max_batch_size),

        'val': generate_(
            val_data,
            generator,
            max_batch_size),

        'test': generate_(
            test_data,
            generator,
            max_batch_size,),
    }

    # save JSON file
    with open('./data/sports/explanation_rational.json', 'w') as file:
        json.dump(new_data, file, indent=2)
        file.close()





if __name__ == "__main__":
    fire.Fire(main)
    # main()
