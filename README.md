# GRAMMAR SCORING ENGINE  

Heres the full walk-thorough of the process:
1. Converting audio speech to text: Using openai-whisper we transcribed the audio files with timestamps of each word i.e start and end timing 
2. Grammar Analysis: Using language_tool_python we checked the transcribed text for any grammatical mistakes i.e spelling errors, punctuation errors 
3. Syntax Analysis: We used Spacy to parse sentence structure i.e Tokenization, POS tagging and Dependency parsing and performed Analysis for clause complexity i.e number of verbs in a sentence, tree depth i.e looking for nested structure in a sentence, subordinate clauses like because, if, when for complexity, sentence completeness and lexical metrics like TTR(Type-Token Ratio, which shows vocabulary richness) and Flesh Reading ease and last analyzing for fluency i.e the usage of words like um, uh, like, you know.
4. Audio Analysis: Using librosa we analyze the sound of speech i.e how fast someone speaks, pausing rate, speaking time ratio, acoustic features(sound quality) and energy/intensity.
5. Machine Learning: Using Xgboost model and the various aforementioned features we train the model to predict the grammar scores from the audio files.
6. Metrics: The metrics used are RMSE, MAE, Accuracy and Pearson correlation

## TRAINING:
The following results were obtained from training the XGBoost model:

`
==================================================
TOP 15 MOST IMPORTANT FEATURES
==================================================
energy_mean         9.745972633361816
error_density       8.338691711425781
unique_word_count   7.800835609436035
pitch_std           6.518698692321777
filler_density      5.836128234863281
total_errors        5.723820209503174
energy_std          5.534993648529053
speaking_time_ratio 5.401168346405029
type_token_ratio    5.36381196975708
max_tree_depth      5.303519248962402
flesch_reading_ease 5.302517890930176
pause_frequency     5.261669158935547
long_pause_count    5.208794116973877
avg_pause_duration  5.125491619110107
word_count          5.092123508453369
==================================================
`
`
==================================================
EVALUATION METRICS
==================================================
RMSE:                    0.7379
MAE:                     0.4814
Accuracy (±0.5):         64.63%
Correlation:             0.5032
==================================================
`

## USAGE 
1. Install the requirements using `pip install -r requirements.txt`
2. Run train to trian.py to train the model 
3. And test.py to test the model or predict label values for audiofiles


