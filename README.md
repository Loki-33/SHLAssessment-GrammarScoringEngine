# GRAMMAR SCORING ENGINE  

Heres the full walk-thorough of the process:
1. Converting audio speech to text: Using openai-whisper we transcribed the audio files with timestamps of each word i.e start and end timing 
2. Grammar Analysis: Using language_tool_python we checked the transcribed text for any grammatical mistakes i.e spelling errors, punctuation errors 
3. Syntax Analysis: We used Spacy to parse sentence structure i.e Tokenization, POS tagging and Dependency parsing and performed Analysis for clause complexity i.e number of verbs in a sentence, tree depth i.e looking for nested structure in a sentence, subordinate clauses like because, if, when for complexity, sentence completeness and lexical metrics like TTR(Type-Token Ratio, which shows vocabulary richness) and Flesh Reading ease and last analyzing for fluency i.e the usage of words like um, uh, like, you know.
4. Audio Analysis: Using librosa we analyze the sound of speech i.e how fast someone speaks, pausing rate, speaking time ratio, acoustic features(sound quality) and energy/intensity.
5. Machine Learning: Using Xgboost model and the various aforementioned features we train the model to predict the grammar scores from the audio files.
6. Metrics: The metrics used are RMSE, MAE, Accuracy and Pearson correlation

## Model Performance

### Top 15 Most Important Features
| Rank | Feature | Importance (Gain) |
|-----:|---------|------------------:|
| 1 | energy_mean | 9.746 |
| 2 | error_density | 8.339 |
| 3 | unique_word_count | 7.801 |
| 4 | pitch_std | 6.519 |
| 5 | filler_density | 5.836 |
| 6 | total_errors | 5.724 |
| 7 | energy_std | 5.535 |
| 8 | speaking_time_ratio | 5.401 |
| 9 | type_token_ratio | 5.364 |
| 10 | max_tree_depth | 5.304 |
| 11 | flesch_reading_ease | 5.303 |
| 12 | pause_frequency | 5.262 |
| 13 | long_pause_count | 5.209 |
| 14 | avg_pause_duration | 5.125 |
| 15 | word_count | 5.092 |

---

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| RMSE | 0.7379 |
| MAE | 0.4814 |
| Accuracy (±0.5) | **64.63%** |
| Pearson Correlation | 0.5032 |



## USAGE 
1. Install the requirements using `pip install -r requirements.txt`
2. Run train to trian.py to train the model 
3. And test.py to test the model or predict label values for audiofiles


