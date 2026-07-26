cd
ls
git add .
git commit -m "Week 2 - Individual python files tested & executed"
git push -u origin main
clear
ls
cd Knowledge_Hands_on/
python src/pipeline.py
python src/pipeline/pipeline.py
mkdir logs
python src/pipeline/pipeline.py
cd logs
clear
cd ..
python -m src.pipeline.pipeline

git add .
git commit -m "Week 2 - Executed pipeline"
git push -u origin main
ls
git status
git add .
git status
clear
cd Knowledge_Hands_on/practice_examples/week03
python streaming_demo.py 
clear
uvicorn practice_examples.week03.streaming_demo:app --reload --port 8002
cd ..
uvicorn practice_examples.week03.streaming_demo:app --reload --port 8002
python streaming_demo.py 
uvicorn practice_examples.week03.streaming_demo:app --reload --port 8002
uvicorn practice_examples.week03.minimal_api:app --reload --port 8001
cd Knowledge_Hands_on/
uvicorn practice_examples.week03.minimal_api:app --reload --port 8001
cd Knowledge_Hands_on/practice_examples/week03
curl -s -X POST localhost:8001/echo  -H 'Content-Type:application/json' -d '{"q":"What is Agentic Ai and how does it helps in Trading?"}'
curl -s -X POST localhost:8001/echo  -H 'Content-Type:application/json' -d '{"quetion":"What is Agentic Ai and how does it helps in Trading?"}'
curl -s -X POST localhost:8001/echo  -H 'Content-Type:application/json' -d '{"question":"What is Agentic Ai and how does it helps in Trading?"}'
cd Knowledge_Hands_on/
uvicorn practice_examples.week03.streaming_demo:app --reload --port 8002
cd Knowledge_Hands_on/practice_examples/week03
curl -s -X POST localhost:8002/ask  -H 'Content-Type:application/json' -d '{"question":"dummy question"}'
curl -s -X POST localhost:8002/ask  -H 'Content-Type:applicatio --no-buffer -X POST -d '{"question":"dummy question"}'
curl -s --no-buffer -X POST localhost:8002/ask  -H 'Content-Type:application/json' -d '{"question":"dummy question"}'
