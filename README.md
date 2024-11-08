# Sample Code for DPO

## Direct Preference Optimization (DPO)

1) SFT
2) DPO
3) Inference

Data set:

The data set in DPO requires a preference score for each pair of responses. For example, if we have a dataset with the following structure:

| Input_Text | Accepted_Completion | Rejected_Completion |
|------------|---------------------|-------------------|
| "What is the capital of France?" | "Paris" | "London" |

we can run DPO to train a LLM to generate responses that are preferred over the rejected ones.

Save the data set in a csv file under `data/`

Check and modify `config.py` for training parameters and locations of models and data.


Then we can run inference with the fine-tuned model. Be sure that the DPO model paths are all correct.

```
python dpo_inference.py -i "What is the square root of 49?"
```


# Usage 

1) First we need to SFT the model.

```
python -m sft
```

Be sure to use the correct model name in the `config.py` file, specifically the `SFTConfig()` dataclass 

2) Then we can use DPO to train the model.

```
python -m dpo
```
Again be sure to use the correct model name in the `config.py` file, specifically the `MyDPOConfig()` dataclass 

3) Then we can run inference with the fine-tuned model.

```
python -m dpo_inference -i "What is the square root of 49?"
```

# Possible Errors:

You might get an error that looks like:

```
policy_output = model.generate(
                    ^^^^^^^^^^^^^^
AttributeError: 'generator' object has no attribute 'generate'
```

This is due to the ever evolving TRL and huggingface APIs. The solution is given here [https://github.com/huggingface/trl/issues/2292](https://github.com/huggingface/trl/issues/2292).
