# Evaluation Metrics
From my research about text evaluation metric, I would like to catagorize them into two main group as following below:

## Traditional 

<details><summary>
  BLEU
</summary>
<br>
  
**BLEU** <img width="138" alt="image" src="https://github.com/RadchaneepornC/LargeLanguageModels/assets/100735165/f9319f48-b5ae-4eae-b85f-53bfd5a76210">
 stands for **B**i**L**ingual **E**valuation **U**nderstudy
- **measures** the similarity of the machine-translated text to a set of high quality reference
- **score** is in the range of [0,1], 1.0 = perfect score
- **formula**

![Alt text](image/BLEU.jpg)


- **example of calculation**

![Alt text](image/Ex_BLEU.jpg)

  
- **tradeoffs**

   - **BLEU is a Corpus-based Metric:** fighting and battling are not be captured as common unigram
   - **normalization and tokenization:** prior to computing the BLEU score, both the reference and candidate translations are normalized and tokenized. The choice of normalization and tokenization steps significantly affect the final BLEU score.


  
- **code for implementation**









</details>

<details><summary>
  ROGUE
</summary></details>
<details><summary>
  METOER
</summary></details>


## Non-traditional 

- ### **Embbeding based**
<details><summary>
  BERTScore
</summary></details>

<details><summary>
 Moverscore
</summary></details>

 - ### **LLM assisted**

<details><summary>
 MTbench
</summary></details>

<details><summary>
Chatbot Arena
</summary></details>


## Reference 
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)
- [Evaluating models](https://cloud.google.com/translate/automl/docs/evaluate)
