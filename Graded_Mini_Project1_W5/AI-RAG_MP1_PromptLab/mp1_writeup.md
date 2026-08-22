Step1:
(i)While validating the job_snippet.jsonl , loading & validation is successful

(ii)While validating the golden_set.jsonl , it is observed that "years_experience_required" is null for last record and the validation will be failed if we declare "years_experience_required:int" with error message thrown which is available in docs/run_output -> "01_goldenset_validation_error.txt"

(iii)To allow years_experience_required: null to be allowed then we need to declare "years_experience_required:int | None = None" if we are manually creating a pipeline.py

(iv) Load the env variables and jobsnippets/goldenset is done.



Step2: 
Define prompt for each type as below,

(i)prompt_zero_shot -> A simple prompt with no defined examples to extract the company name, job role, and number of years of experience from a job description.

(ii)prompt_few_shot -> A  prompt to extract the company name, job role, and number of years of experience from a job description but with an example provided.

(iii)prompt_structured ->  A  prompt to extract the company name, job role, and number of years of experience from a job description but with an example, tasks, output format with a clear structure to get the required output.

(iv)prompt_cot -> A  prompt to extract the company name, job role, and number of years of experience from a job description with reasoning rules to get better results on more complex tasks that require reasoning before responding.



Step3:
(i)To test running one strategy(zero shot) either one snippet ID or all snippet by calling,
results = await run_one("zero_shot", snippets[0])
results = await run_one("zero_shot", snippets)

(ii) I used Zero shot prompt for testing purpose and called run_one method to run each snippet against zero prompt to get the output. Rather than making each call sequentially , use gather sync logic which in turn call the sub function "run_each_snippet" to make gpt-4o-mini LLM call to getch the output.Use parse function to parse the respone to get company, role, years of experience and in addition to that calculate the cost & latency using the standard calculation logic. While calculating the cost with (prompt_tokens * in_rate + completion_tokens * out_rate), the cost result will be in exponential format and to get the desired output, use ${final_cost:.8f) to see it in proper format.

(iii) To run all 10 snippets against 4 prompts to make 4*10 calls, use results = await run_all(snippets). The response is returned and used for further and also written to output file.

(iv) Pass the snippet argument as list because the snippets is a list in async def run_all()  function.


Step4:
(i)To calculate the accuracy of the result, extract the company, role, years_of_experience and match against the golden set.If matched, set the increase the score by 1.
(ii) Create a score_llm_judge function to make gpt-4 LLM call using RUBRIC evaluation logic thereby passing appropriate prompt with extracted resultset,context to obtain the responses.
(iii) The output of the score_llm_judge will be JudgeVerdict as it cannot be single integer as the async function receives those four inputs and returns a JudgeVerdict object.
(iv) if extracted is None:
        return JudgeVerdict(
            accuracy=1,
            groundedness=1,
            format_score=1,
            reasoning="The extracted response is missing or could not be parsed."
        ) because pydantic is returing 1


I have run the code three times and the outputs are stored in txt file all the three times as below.

First attempt:
runall_batch -> output3.txt
judge_response -> judge_response.txt

Second attempt:
runall_batch -> output4.txt
judge_response -> judge_response1.txt

Third attempt:
runall_batch -> output5.txt
judge_response -> judge_response2.txt
