I have run the code three times and below is the ouput.

First attempt:

strategy   Accuracy (mean of 3)	Parse rate	Judge score	Total cost ($)	Latency p50 (s)
					
cot	        2.8	                1.0         3.9	         0.001	        1.262
few_shot	2.7	                1.0	        3.9	         0.000	        1.029
structured	2.7	                1.0	        3.9	         0.001	        1.005
zero_shot	2.6	                1.0	        3.9	         0.000	        1.140


Second attempt:

strategy   Accuracy (mean of 3)	Parse rate	Judge score	Total cost ($)	Latency p50 (s)
					
cot	        2.8	                1.0         4.0	         0.001	        1.177
few_shot	2.7	                1.0	        3.9	         0.000	        0.984
structured	2.7	                1.0	        3.9	         0.001	        1.141
zero_shot	2.6	                1.0	        4.0	         0.000	        1.191



Third attempt:

strategy   Accuracy (mean of 3)	Parse rate	Judge score	Total cost ($)	Latency p50 (s)
					
cot	        2.8	                1.0         4.0	         0.001	        1.154
few_shot	2.7	                1.0	        3.9	         0.000	        1.034
structured	2.7	                1.0	        3.9	         0.001	        1.087
zero_shot	2.6	                1.0	        4.0	         0.000	        1.110
