"""GPU regression for prompt-logprob row order, values and bounded LM-head batches.

Run with the patched venv: python bench/test_prompt_logprob_chunk_size.py
"""
import torch
import vllm.v1.worker.gpu.sample.prompt_logprob as worker

if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU required')

torch.manual_seed(42)
torch.set_float32_matmul_precision('highest')
hidden=torch.randn(37,32,device='cuda',dtype=torch.float32)
weight=torch.randn(32,257,device='cuda',dtype=torch.float32)
targets=torch.arange(37,device='cuda')*7%257
original=worker.CHUNK_SIZE

try:
    with torch.inference_mode():
        for topk in [0,3]:
            calls=[]
            def logits(rows):
                calls.append(len(rows))
                return rows@weight
            worker.CHUNK_SIZE=1024
            reference=worker.compute_prompt_logprobs_with_chunking(targets,hidden,logits,topk)
            assert calls==[37]
            calls.clear()
            worker.CHUNK_SIZE=7
            result=worker.compute_prompt_logprobs_with_chunking(targets,hidden,logits,topk)
            assert max(calls)<=7 and sum(calls)==37,calls
            for actual,expected in zip(result,reference):
                if actual.dtype.is_floating_point:
                    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
                else:
                    assert torch.equal(actual,expected)
            print(f'PASS topk={topk}: all 37 rows preserved, maximum LM-head batch {max(calls)}')
finally:
    worker.CHUNK_SIZE=original
