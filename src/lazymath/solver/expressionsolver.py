from vllm import LLM, SamplingParams

class ExpressionSolver:
    def __init__(self, model="Qwen/Qwen2.5-Math-1.5B"):
        self.model = model 
        self.sampling_params = SamplingParams()
        self.llm = LLM(model=model,dtype="float16")

    def solve_expression(self, expression: str, task: str = "Solve expression "):

        outputs = self.llm.generate(task+expression, self.sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
