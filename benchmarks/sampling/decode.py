"""Fixed Ling rollout benchmark; run baseline/candidate in separate processes."""
import argparse
import json
import time
from pathlib import Path
from areno import Trainer
from areno.api import CudaConfig, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=8)
    args=parser.parse_args()
    trainer=Trainer(args.tp,args.model, custom_config=CudaConfig(tp_size=args.tp,
            devices=list(range(args.tp)), runtime={'compile_model':False,'eager_decode':args.eager,
                                                    'decode_graph_buckets':[1,8]}))
    trainer.init()
    trainer.begin_rollout_session()
    results=[]
    try:
        prompts=['Explain why the sky is blue in simple words.'] * args.batch
        for k,p in ((50,.9),(0,.9),(0,1.)):
            # Fit total latency vs output length to separate prefill overhead.
            record={'k':k,'p':p,'tp':args.tp,'batch':args.batch,'eager':args.eager,'lengths':[]}
            for length in (16,64):
                params=SamplingParams(top_k=k,top_p=p,temperature=.8,max_new_tokens=length,ignore_eos=True,seed=634)
                trainer.rollout_batch(prompts,1,params)
                samples=[]
                for _ in range(3):
                    start=time.perf_counter(); out=trainer.rollout_batch(prompts,1,params)
                    elapsed=time.perf_counter()-start
                    assert all(len(r.sequences[0].resp_tokens)==length for r in out)
                    samples.append(elapsed)
                record['lengths'].append({'tokens':length,'seconds':samples})
            a,b=record['lengths']
            slope=(sorted(b['seconds'])[1]-sorted(a['seconds'])[1])/48
            record['incremental_tpot_ms']=1000*slope
            record['incremental_tokens_per_second']=args.batch/slope
            results.append(record)
            Path(args.output).write_text(json.dumps(results,indent=2))
            print('DECODE634',json.dumps(record),flush=True)
    finally:
        trainer.end_rollout_session(); trainer.close()


if __name__=='__main__': main()
