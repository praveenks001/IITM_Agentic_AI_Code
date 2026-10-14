Step 1: 

Made the environment ready by changing the Qdrant credentials in .env file
Installed required dependencies for Qdrant

Step 2:

(i) load_corpus -  To consume the Qdrant details from os , we need to load dotenv by adding load_dotenv() along with its import 

from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"

Once this step is done, the qdrant keys are loaded in the environment

(ii) chunk_document - As mentioned in the requirement: In cmd_ingest function, each document from the corpus will be iterated and passed to chunk_document function where in turn it will call another function called "chunk_text" to create chunks of 500 chars with an overlap of 40 characters using "Sliding window chunking" method and assign a ID to each chunk. Like this, all 5 documents are chunked and total chunks are below, 
  01_red_headed_league.txt: 12 chunks
  02_speckled_band.txt: 13 chunks
  03_blue_carbuncle.txt: 12 chunks
  04_engineers_thumb.txt: 14 chunks
  05_scandal_in_bohemia.txt: 16 chunks
→ Total chunks: 67


(iii) setup_collection  - Calling the qdrant collection creation function to create "mp2_sherlock" collection. If its already created, it will just return the collection name and if not, it will create the collection with 1536 dims and return the collection.


(iv) embed_texts  - Using Embbedding model, Each chunk of the corpus will be embedded passing into this method. Create a function called is_qdrant_collection_populated in order to check if the qdrant is available with embedded chunks in it else no need to embed again.


(v) ingest_chunks - if quadrant is already present dont do anything as it can contains embedded chunks already. If not, we can embed the chunks and upsert them into the collection.

(vi) retrieve - Since we are using qdrant, the retrieve function has isQdrant check , if yes - we can retrieve the cosine similarity chunks from qdrant collection for each golden set question from jsonl files returning from validateagainst method.

(vii) answer - once the cosine similarity chunks are retrieved, build a prompt using build prompt method where question, retrieved value, system prompt will be given to create a system, user_msg. Now call the chat model (gpt-4o-mini) to get the answers withe calculated / defined paramters like cost, latency, tokens_in, tokens_out, citations etc.


Below are the random questions i have asked by callig python mp2_rag.py ask .

Q1: who is sherlock holmes?
Q2: who is the detective in this story?
These two questions are easy questions and got expected output as below.


Qdrant Keys loaded
Qdrant is::: <qdrant_client.qdrant_client.QdrantClient object at 0x7f7c2bd93670>
calling main
Executing main:::
sys.argv is::: ['mp2_rag.py', 'ask']
calling main cmd ask
Mini-RAG over the Sherlock Holmes corpus.
Type your question. Empty line or Ctrl-C to exit.

? who is sherlock holmes?
Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks

Sherlock Holmes is a fictional detective created by Sir Arthur Conan Doyle. He is known for his keen observation skills, logical reasoning, and use of forensic science to solve complex cases. In the context of the stories, he is often referred to as "THE woman" by Watson, indicating his unique relationship with Irene Adler, a key character in "A Scandal in Bohemia" (source: A Scandal in Bohemia, section 0).

  Sources:
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#5
    - 04_engineers_thumb.txt — 04_engineers_thumb.txt#10
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#0
  Latency: 2954.3142870097654ms

?who is the detective in this story?
Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks

The detective in this story is Sherlock Holmes. (The Engineer's Thumb + Investigation)

  Sources:
    - 04_engineers_thumb.txt — 04_engineers_thumb.txt#10
    - 04_engineers_thumb.txt — 04_engineers_thumb.txt#2
    - 03_blue_carbuncle.txt — 03_blue_carbuncle.txt#11
  Latency: 1748.228640994057ms


Q3: is sherlock holmes married?
The details are not mentioned in the document, looks like it assumed this context and answered as no.

? 
No, Sherlock Holmes is not married. The excerpts do not indicate that he has a wife or is in a marital relationship. In fact, the mention of Holmes being involved in a marriage ceremony is related to his role as a witness, not as a participant. (Source: "A Scandal in Bohemia" + section 7)

  Sources:
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#15
    - 04_engineers_thumb.txt — 04_engineers_thumb.txt#2
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#7
  Latency: 2454.130213998724ms


Q4:is sherlock holmes an african?

When i ask the same out of document question, it answers clearly that the information is not available.

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks

The excerpts provided do not contain any information regarding Sherlock Holmes's ethnicity or background. Therefore, I cannot answer your question based on the provided texts.

  Sources:
    - 01_red_headed_league.txt — 01_red_headed_league.txt#7
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#2
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#6
  Latency: 1997.1956049994333ms

Q5: When did the sherlock holmes die and who are all attended the mournings?

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks

The excerpts provided do not contain any information about the death of Sherlock Holmes or who attended the mournings.

  Sources:
    - 02_speckled_band.txt — 02_speckled_band.txt#6
    - 05_scandal_in_bohemia.txt — 05_scandal_in_bohemia.txt#12
    - 02_speckled_band.txt — 02_speckled_band.txt#7
  Latency: 1811.588329001097ms

? 



Step 3/4: python mp2_rag.py validate

Qdrant Keys loaded
Qdrant is::: <qdrant_client.qdrant_client.QdrantClient object at 0x7f494973b670>
calling main
Executing main:::
sys.argv is::: ['mp2_rag.py', 'validate']
calling main cmd validate

  Validating 2 questions from predefined_questions.jsonl…

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks
The answer for below question is:: The assistant working at Jabez Wilson's pawnshop was named Spaulding. His real identity, however, is not explicitly mentioned in the provided excerpts. Therefore, I cannot provide that information. (The Red-Headed League + section 5)
  ✓ q1_red_headed_league_assistant
      Q: Who was the assistant working at Jabez Wilson's pawnshop, and what was his real identity?
      Cited: 01_red_headed_league.txt
      Expected: 01_red_headed_league.txt
      Facts matched: 0/5
      Latency: 2433.8899979920825ms

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks
The answer for below question is:: The excerpts do not provide a clear answer regarding what specifically killed Julia Stoner at Stoke Moran or how the method was discovered. They mention that Julia cried out about "the band" and that she died without any apparent cause, but they do not detail the method of her death or how it was uncovered. Therefore, I cannot provide an answer based on the provided excerpts. (The Adventure of the Speckled Band)
  ✓ q2_speckled_band_murder_weapon
      Q: What killed Julia Stoner at Stoke Moran, and how was the method discovered?
      Cited: 02_speckled_band.txt
      Expected: 02_speckled_band.txt
      Facts matched: 0/6
      Latency: 2270.858132003923ms

  Source-match: 2/2


  Validating 3 questions from learner_questions.jsonl…

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks
The answer for below question is:: The Blue Carbuncle was discovered by Peterson's wife, who found it embedded in the bird's crop. (The Adventure of the Blue Carbuncle, section 3)
  ✓ q3_blue_carbuncle
      Q: Who discovered the Blue Carbuncle inside the goose, and where in the bird was it found?
      Cited: 03_blue_carbuncle.txt
      Expected: 03_blue_carbuncle.txt
      Facts matched: 3/4
      Latency: 2014.315558000817ms

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks
The answer for below question is:: Victor Hatherley realized that Colonel Stark's hydraulic press was not being used for fuller's earth because the floor of the press chamber was marked by a metallic crust rather than the residue of earth. This indicated that the press had been used to compress metal, specifically to stamp half-crowns from a base alloy of nickel and tin. The "fuller's earth" cover story was a ruse to explain the press's presence without raising suspicion (source: "The Engineer's Thumb" + sections 6 and 12).
  ✓ q4_engineers_thumb
      Q: What evidence showed Victor Hatherley that Colonel Stark's hydraulic press was not being used for fuller's earth, and what was its actual purpose?
      Cited: 04_engineers_thumb.txt
      Expected: 04_engineers_thumb.txt
      Facts matched: 2/6
      Latency: 2365.5427080084337ms

Qdrant collection points: 67
Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks
The answer for below question is:: Irene Adler recognized Sherlock Holmes in his groom disguise and again in the clergyman disguise when he was carried into her house. She was aware of his presence and the fire trick he employed was an old tactic that she was familiar with. As for the compromising photograph, she retained it as a safeguard in case the King ever tried to harm her or her husband (source: "A Scandal in Bohemia" + sections 12 and 14).
  ✓ q5_scandal_bohemia
      Q: How did Irene Adler recognize and outmaneuver Sherlock Holmes despite his disguises and fire ruse, and why did she retain the compromising photograph after leaving England?
      Cited: 05_scandal_in_bohemia.txt
      Expected: 05_scandal_in_bohemia.txt
      Facts matched: 0/7
      Latency: 2793.7014769995585ms

  Source-match: 3/3

The output is also saved into validate_output.txt file

  Step 5: is sherlock holmes married?
  For this question, the model anwered as No and then it said there is no finding for his marriage evidence. Ideally it should have told the context is not availabe in the given document / corpus as similar to Q4, Q5 in step 2