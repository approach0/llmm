from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.3, top_p=0.95)

llm = LLM(model="WizardLM/WizardMath-13B-V1.0")
print('model loaded.')

for _ in range(3):
    outputs = llm.generate(['math is a '], sampling_params, use_tqdm=False)
    print(outputs[0].outputs[0].text)
