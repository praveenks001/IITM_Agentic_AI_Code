Step 1: 

Made the environment ready by changing the Qdrant credentials in .env file
Installed required dependencies for Qdrant

(i)To consume the Qdrant details from os , we need to load dotenv by adding load_dotenv() along with its import 

from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"

Once this step is done, the qdrant keys are loaded in the environment

-----------------------------------------------------------------------------------------------------

Step 2: Implement the four functions

(i) format_context: Since it is asked to return context (from "<title>" — <section>) , sources (n, source, title, section, score) the chunking strategy has already n, source, title, section so iterate the chunks and map the respective values to return it. For score, the default value is set to 0

(ii) validate_citations: Since it is asked to create and return used, invalid, has_citations and the logicas are given in the description, so written the function in same way

(iii) should_abstain: The function is created with a logic where if there are no chunks, the value will be True and if it is less than the threshold (0.30) then it is False;

(iv) evaluate: Contains the final summary to determine , was the question answerable, abstained, retrieval hit rate, ground rate, accuracy, hallucination rate.

(v) python mp3_grounded_qa.py selftest :
  ✓ should_abstain: low score → abstain
  ✓ should_abstain: good score → answer
  ✓ should_abstain: no chunks → abstain
  ✓ evaluate: hit rate 1/2 answerable
  ✓ evaluate: grounded rate 1/2 answerable
  ✓ evaluate: abstention accuracy 1/1
  ✓ evaluate: hallucination rate 0/1
  ✓ format_eval_table renders a table

✓ selftest passed


-----------------------------------------------------------------------------------------------------

Step 3: python mp3_grounded_qa.py ingest:

(i) The provided corpus has been loaded.
(ii) Each documents will be chunked using "Paragraph aware chunking" strategy
(iii) A collection name called "mp3_sherlock" will be created in Qdrant and the embedded chunks will be upserted.

Qdrant Keys loaded
→ Loading corpus…
  01_red_headed_league.txt: 13 chunks
  02_speckled_band.txt: 13 chunks
  03_blue_carbuncle.txt: 13 chunks
  04_engineers_thumb.txt: 16 chunks
  05_scandal_in_bohemia.txt: 18 chunks
→ 73 chunks · setting up collection · ingesting…
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:151: DeprecationWarning: `recreate_collection` method is deprecated and will be removed in the future. Use `collection_exists` to check collection existence and `create_collection` instead.
  qdrant.recreate_collection(
✓ Done. Try: python mp3_grounded_qa.py ask


python mp3_grounded_qa.py ask : 

Answers are retrieved from the document if its available as below.


Qdrant Keys loaded
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:83: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.19.0. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
  qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
Grounded Q&A over the Sherlock corpus. Empty line to exit.

? What time the peterson walked to home through Tottenham court?
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:166: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
  hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=k)

Peterson walked home through Tottenham Court Road at around four o'clock on Christmas morning [1].

  Sources:
    [1] The Adventure of the Blue Carbuncle — The Hat and the Goose  (score 0.528)
  Latency: 2108ms

-------------------------
If answers are not available in the document then it will say I don't know based on the provided documents.

How Praveen is invloved in the scandal?

I don't know based on the provided documents.

  (abstained — no supporting excerpt)
  Latency: 1624ms


Who is Professor Moriarty?

I don't know based on the provided documents.

  (abstained — no supporting excerpt)
  Latency: 1758ms


-----------------------------------------------------------------------------------------------------

Step 4:
 python mp3_grounded_qa.py eval 


Qdrant Keys loaded
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:83: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.19.0. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
  qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:166: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
  hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=k)

  id                              type     hit   abstained  grounded  ok
  --------------------------------------------------------------------------
  q1_red_headed_league_assistant  answerable ✓     ✓          ✗         ✗
  q2_speckled_band_weapon         answerable ✓     ✓          ✗         ✗
  q3_scandal_bohemia_adler        answerable ✓     ✗          ✓         ✓
  q4_engineers_thumb_press        answerable ✓     ✗          ✓         ✓
  abstain_moriarty                abstain  ✗     ✓          ✗         ✓
  abstain_out_of_domain           abstain  ✗     ✓          ✗         ✓
  --------------------------------------------------------------------------

  Retrieval hit rate    : 1.0   (4 answerable questions)
  Grounded & cited rate : 0.5   (KPI: grounded response rate)
  Abstention accuracy   : 1.0   (2 should-abstain questions)
  Hallucination rate    : 0.0   (KPI: hallucination rate)


Step 5: Added 3 more questions in which 2 are answerable and 1 is abstain in learner_Questions.jsonl and re-run the eval command.

Qdrant Keys loaded
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:83: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.19.0. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
  qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:166: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
  hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=k)

  id                              type     hit   abstained  grounded  ok
  --------------------------------------------------------------------------
  q1_red_headed_league_assistant  answerable ✓     ✓          ✗         ✗
  q2_speckled_band_weapon         answerable ✓     ✓          ✗         ✗
  q3_scandal_bohemia_adler        answerable ✓     ✗          ✓         ✓
  q4_engineers_thumb_press        answerable ✓     ✗          ✓         ✓
  abstain_moriarty                abstain  ✗     ✓          ✗         ✓
  abstain_out_of_domain           abstain  ✗     ✓          ✗         ✓
  q5_blue_carbuncle_discovery     answerable ✓     ✗          ✓         ✓
  q6_scandal_bohemia_escape       answerable ✓     ✗          ✓         ✓
  q7_abstain_hound_baskervilles   abstain  ✗     ✓          ✗         ✓
  --------------------------------------------------------------------------

  Retrieval hit rate    : 1.0   (6 answerable questions)
  Grounded & cited rate : 0.6666666666666666   (KPI: grounded response rate)
  Abstention accuracy   : 1.0   (3 should-abstain questions)
  Hallucination rate    : 0.0   (KPI: hallucination rate)

if Thresold is set to 0.60 then below is the result,

Qdrant Keys loaded
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:83: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.19.0. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
  qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:166: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
  hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=k)

  id                              type     hit   abstained  grounded  ok
  --------------------------------------------------------------------------
  q1_red_headed_league_assistant  answerable ✓     ✓          ✗         ✗
  q2_speckled_band_weapon         answerable ✓     ✓          ✗         ✗
  q3_scandal_bohemia_adler        answerable ✓     ✓          ✗         ✗
  q4_engineers_thumb_press        answerable ✓     ✓          ✗         ✗
  abstain_moriarty                abstain  ✗     ✓          ✗         ✓
  abstain_out_of_domain           abstain  ✗     ✓          ✗         ✓
  q5_blue_carbuncle_discovery     answerable ✓     ✗          ✓         ✓
  q6_scandal_bohemia_escape       answerable ✓     ✗          ✓         ✓
  q7_abstain_hound_baskervilles   abstain  ✗     ✓          ✗         ✓
  --------------------------------------------------------------------------

  Retrieval hit rate    : 1.0   (6 answerable questions)
  Grounded & cited rate : 0.3333333333333333   (KPI: grounded response rate)
  Abstention accuracy   : 1.0   (3 should-abstain questions)
  Hallucination rate    : 0.0   (KPI: hallucination rate)
  

if Threshold is set to 0.95 then below is the result,

Qdrant Keys loaded
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:83: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.19.0. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
  qdrant = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"),
/voc/work/Week 12_ Graded Mini Project Documents/AI-RAG_MP3_GroundedQA/learner/mp3_grounded_qa.py:166: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
  hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=k)

  id                              type     hit   abstained  grounded  ok
  --------------------------------------------------------------------------
  q1_red_headed_league_assistant  answerable ✓     ✓          ✗         ✗
  q2_speckled_band_weapon         answerable ✓     ✓          ✗         ✗
  q3_scandal_bohemia_adler        answerable ✓     ✓          ✗         ✗
  q4_engineers_thumb_press        answerable ✓     ✓          ✗         ✗
  abstain_moriarty                abstain  ✗     ✓          ✗         ✓
  abstain_out_of_domain           abstain  ✗     ✓          ✗         ✓
  q5_blue_carbuncle_discovery     answerable ✓     ✓          ✗         ✗
  q6_scandal_bohemia_escape       answerable ✓     ✓          ✗         ✗
  q7_abstain_hound_baskervilles   abstain  ✗     ✓          ✗         ✓
  --------------------------------------------------------------------------

  Retrieval hit rate    : 1.0   (6 answerable questions)
  Grounded & cited rate : 0.0   (KPI: grounded response rate)
  Abstention accuracy   : 1.0   (3 should-abstain questions)
  Hallucination rate    : 0.0   (KPI: hallucination rate)