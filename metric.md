# Edit distance
- Reference: https://en.wikipedia.org/wiki/Edit_distance
- Edit distance is a way of quantifying how dissimilar two strings (e.g., words) are to one another by counting the minimum number of operations required to transform one string into the other.
Types of edit distance
  - Levenshtein distance allows deletion, insertion and substitution.

# Error Rate (CER (Character Error Rate), WER (Word Error Rate))
## CER
- CER calculation is based on the concept of Levenshtein distance, where we count the minimum number of character-level operations required to transform the ground truth text (aka reference text) into the OCR output.
- CER = (S + D + I) / N
  - N: Number of characters in grund truth (= S + D + C)
```python
import import jiwer

cer = jiwer.cer(gt, pred)
wer = jiwer.wer(gt, pred)
```