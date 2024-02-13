# RDRec

## Step. 1 distill rationale before running RDRec

#### (a) Install llama 2 （download model weights and tokenizer）
        get the License from [the site](https://llama.meta.com/llama-downloads/)
        >> cd llama 
	    >> ./download.sh (License required)
        >> pip install -e .

#### (b) Test llama 2 environment  (under ./llama )
        >> torchrun --nproc_per_node 1 example_chat_completion.py \
          --ckpt_dir llama-2-7b-chat/ \
          --tokenizer_path tokenizer.model \
          --max_seq_len 512 --max_batch_size 6

#### (c) Rationale distillation  ({dataset}: beauty, sports, and toys.) (under ./RDRec )
        >> torchrun --nproc_per_node 1 data/{dataset}/distillation_{dataset}.py \
          --ckpt_dir llama/llama-2-7b-chat/ \
          --tokenizer_path llama/tokenizer.model \
          --max_seq_len 512 --max_batch_size 6

## Step. 2 train and test RDRec

#### (a) Install requirement 
        >> pip install -r  requirement.txt

#### (b) Pre-training 
        >> python pretrain.py

#### (c) Recommendation inference
        >> python seq.py
        >> python topn.py
        >> python exp.py




## Amazon Data & Model Checkpoint to [Download](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/Eq-8HUTFas1Fm0xw2-4S-9IBGmRzW2GGA-ZJi2d3Q2HxTQ?e=vp7Iiy)
- Sports & Outdoors
- Beauty
- Toys & Games

## Code Dependencies
- Python 3.6
- PyTorch 1.6
- transformers 4.18.0

## Code Reference
- [P5](https://github.com/jeykigung/P5)
- [POD](https://github.com/lileipisces/POD)
- [llama 2](https://github.com/facebookresearch/llama)

