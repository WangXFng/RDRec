# RDRec [arXiv](https://arxiv.org/pdf/2405.10587)

## Paper
- RDRec: Rationale Distillation for LLM-based Recommendation, **ACL 2024 Main (short)**.
- [**Xinfeng Wang**](https://wangxfng.github.io/), Jin Cui, Yoshimi Suzuki, Fumiyo Fukumoto.

## Note
- Please use the latest code released on **<u>June 11th, 2024</u>**.
- The checkpoints of the RDRec model were uploaded on [Google Drive](https://drive.google.com/drive/folders/1bwhliM4KN8pBdk5c0pRPDVCgTJbeOk0s) and [Baidu Drive](https://pan.baidu.com/s/15TQ6zi-ZHfPik02bjlPwRQ?pwd=eb3d ).
- The experimental setup follows [POD](https://github.com/lileipisces/POD). If there is any problem, please check our code or this paper.


## Instruction
### Step. 1 distill rationale before running RDRec

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

### Step. 2 train and test RDRec

#### (a) Install requirement 
        >> pip install -r  requirement.txt

#### (b) Pre-training ({dataset}: beauty, sports, and toys.) (under ./RDRec )
        >> python pretrain.py ./data/{dataset}/ --cuda --batch_size 64 --checkpoint ./checkpoint/{dataset}/

#### (c) Recommendation inference 
        >> python seq.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/
        >> python topn.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/
        >> python exp.py ./data/{dataset}/ --cuda --batch_size 32 --checkpoint ./checkpoint/{dataset}/


## Others
- All experiments, including rationale distillation, can be conducted on a **<u>single Nvidia GeForce RTX 3090 (24GB memory)</u>**. Reduce the batch size if you encounter an OOM error on some dataset.
- There are some fluctuations in RDRec's results for sequential recommendations. We reported average results in 10-trial runs in the paper  (See [t_test.py](https://github.com/WangXFng/RDRec/blob/main/utils/t_test.py) for more details). If the results are not ideal, please pre-train the model once again. 
- If you have any questions, please feel free to contact me at kaysenn@163.com.


## Code Reference
- [P5](https://github.com/jeykigung/P5)
- [POD](https://github.com/lileipisces/POD)
- [llama 2](https://github.com/facebookresearch/llama)


## Citation
If this repository helps you, please cite:

	@inproceedings{wang2024rdrec,
	  title={RDRec: Rationale Distillation for LLM-based Recommendation},
	  author={Wang, Xinfeng and Cui, Jin and Suzuki, Yoshimi and Fukumoto, Fumiyo},
	  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
	  year={2024}
	}
