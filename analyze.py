import language_tool_python
import spacy 
import numpy as np 
import textstat
import librosa


class GrammarAnalyzer:
    '''
        Evaluating Syntax complexities like 
        sentence length- the longer the sentence the more complex it is 
        clauses- a sentence having more clauses will be more complex
        subordinate clauses- words like because, when, if these add complexity to the sentences 
        depth- it measures how nested a senteces it 
        sentece completeness - checks if the sentences have proper structure
        Type-Token Ratio - measures richness of vocabulary
        Flesh Reading Ease - evaluates reading easiness like Very easy(5th grade), difficult(college level)
        Fluency- usage of words like um, like, uh, sort, heavy usage of such words represents dullness 
    '''
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')
        self.nlp = spacy.load('en_core_web_lg')
    
    def analyze_grammar(self, text):
        features = {}

        matches = self.tool.check(text)
        features.update(self._extract_grammar_errors(matches, text))

        doc = self.nlp(text)
        features.update(self._extract_syntax_features(doc))
        features.update(self._extract_completeness_features(doc))

        features.update(self._extract_text_stats(text))

        features.update(self._extract_fluency_markers(text))

        return features

    def _extract_grammar_errors(self, matches, text):
        '''
        Extracting various grammatical, spelling, pronunciation errors
        '''

        word_count = len(text.split())

       
        error_types = {
            'grammar': 0,
            'spelling': 0,
            'punctuation': 0,
            'typography': 0,
            'style': 0
        }

        for match in matches:
            category = match.category.lower()
            for error_type in error_types:
                if error_type in category:
                    error_types[error_type] += 1 
                    break 

        return {
            'total_errors': len(matches),
            'error_density': (len(matches) / word_count * 100) if word_count > 0 else 0,
            'grammar_errors': error_types['grammar'],
            'spelling_errors': error_types['spelling'],
            'punctuation_errors': error_types['punctuation'],
            'typography_errors': error_types['typography'],
            'style_errors': error_types['style']
        }


    
    def _extract_syntax_features(self, doc):
     
        sentences = list(doc.sents)

        if len(sentences) == 0:
            return self._empty_syntax_features()

        sent_lengths = [len(sent) for sent in sentences]
        clause_counts = []
        for sent in sentences:
            verbs = [token for token in sent if token.pos_ == 'VERB']
            clause_counts.append(len(verbs))

        tree_depths = []
        for sent in sentences:
            max_depth = 0
            for token in sent:
                depth = self._get_tree_depth(token)
                max_depth = max(max_depth, depth)
            tree_depths.append(max_depth)


        sub_conjunctions = len([token for token in doc if token.dep_ == 'mark'])

        coord_conjunctions = len([token for token in doc if token.pos_ == 'CCONJ'])

        return {
            'num_sentences': len(sentences),
            'avg_sentence_length': np.mean(sent_lengths),
            'max_sentence_length': np.max(sent_lengths),
            'min_sentence_length': np.min(sent_lengths),
            'std_sentence_length': np.std(sent_lengths),
            'avg_clauses_per_sentence': np.mean(clause_counts),
            'avg_tree_depth': np.mean(tree_depths),
            'max_tree_depth': np.max(tree_depths),
            'subordinate_conjunctions': sub_conjunctions,
            'coordinating_conjunctions': coord_conjunctions,
            'subordinate_clause_ratio': sub_conjunctions / len(sentences) if len(sentences) > 0 else 0
        }

        
    def _get_tree_depth(self, token):
        depth = 0
        current = token 
        while current.head != current:
            depth += 1 
            current = current.head 
        return depth 

    def _extract_completeness_features(self, doc):
        sentences = list(doc.sents)

        incomplete_count = 0
        for sent in sentences:
            has_subject = any(token.dep_ in ['nsubj', 'nsubj[ass'] for token in sent)
            has_verb = any(token.pos_ == 'VERB' for token in sent)
            
            if not (has_subject and has_verb):
                incomplete_count += 1 
        return {
            'incomplete_sentences': incomplete_count,
            'complete_sentence_ratio': (len(sentences) - incomplete_count) / len(sentences) if len(sentences) > 0 else 0
        }

    def _extract_text_stats(self, text):
        words = text.split()
        unique_words = set(words)
        
        return {
            'word_count': len(words),
            'unique_word_count': len(unique_words),
            'type_token_ratio': len(unique_words) / len(words) if len(words) > 0 else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if len(words) > 0 else 0,
            'flesch_reading_ease': textstat.flesch_reading_ease(text) if len(text) > 10 else 0,
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text) if len(text) > 10 else 0
        }

    def _extract_fluency_markers(self, text):
        text_lower = text.lower()
        
        # Filler words
        fillers = ['um', 'uh', 'like', 'you know', 'i mean', 'sort of', 'kind of']
        filler_count = sum(text_lower.count(filler) for filler in fillers)
        
        # Repetitions
        words = text.split()
        repetitions = sum(1 for i in range(len(words)-1) if words[i].lower() == words[i+1].lower())
        
        return {
            'filler_words': filler_count,
            'repetitions': repetitions,
            'filler_density': (filler_count / len(words) * 100) if len(words) > 0 else 0
        }
    
    def _empty_syntax_features(self):
        return {
            'num_sentences': 0,
            'avg_sentence_length': 0,
            'max_sentence_length': 0,
            'min_sentence_length': 0,
            'std_sentence_length': 0,
            'avg_clauses_per_sentence': 0,
            'avg_tree_depth': 0,
            'max_tree_depth': 0,
            'subordinate_conjunctions': 0,
            'coordinating_conjunctions': 0,
            'subordinate_clause_ratio': 0
        }
    
    
class AudioAnalyzer:
    """
    Analyzing teh sound of the speech using the following features
    Speech Rate(wpm) - how fast someone speaks 
    Pause analysis - how long the pauses are 
    Speaking Time Ratio - gives us average time spent speaking instead of keeping silent 
    Pitch vairation - analyzes various pitch used while speaking, evaluatinf for expressiveness 
    Energy/ Intensity - represents consistency, engagement(vairable energy) and uncertainess(very low energy) 
    """
    def __init__(self):
        pass
    
    def analyze_audio(self, audio_path, segments):
        """Extract audio-based features"""
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        word_times = []
        for segment in segments:
            if 'words' in segment:
                for word in segment['words']:
                    word_times.append({
                        'start': word['start'],
                        'end': word['end'],
                        'word': word['word']
                    })
        
        features = {}
        
        word_count = len(word_times)
        features['speech_rate_wpm'] = (word_count / duration * 60) if duration > 0 else 0
        
        if len(word_times) > 1:
            pauses = []
            for i in range(len(word_times) - 1):
                pause = word_times[i+1]['start'] - word_times[i]['end']
                if pause > 0:
                    pauses.append(pause)
            
            if len(pauses) > 0:
                features['avg_pause_duration'] = np.mean(pauses)
                features['max_pause_duration'] = np.max(pauses)
                features['pause_frequency'] = len(pauses) / duration if duration > 0 else 0
                features['std_pause_duration'] = np.std(pauses)
                
                # Long pauses (>0.5s) 
                long_pauses = [p for p in pauses if p > 0.5]
                features['long_pause_count'] = len(long_pauses)
                features['long_pause_ratio'] = len(long_pauses) / len(pauses) if len(pauses) > 0 else 0
            else:
                features.update(self._empty_pause_features())
        else:
            features.update(self._empty_pause_features())
            features['speech_rate_wpm'] = 0
        
        speaking_time = sum([wt['end'] - wt['start'] for wt in word_times])
        features['speaking_time_ratio'] = speaking_time / duration if duration > 0 else 0
        
        features.update(self._extract_acoustic_features(y, sr))
        
        return features
    
    def _extract_acoustic_features(self, y, sr):
        """Extract pitch and energy features"""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        rms = librosa.feature.rms(y=y)[0]
        
        return {
            'pitch_mean': np.mean(pitch_values) if len(pitch_values) > 0 else 0,
            'pitch_std': np.std(pitch_values) if len(pitch_values) > 0 else 0,
            'pitch_range': (np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 0 else 0,
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms)
        }
    
    def _empty_pause_features(self):
        """Return empty pause features"""
        return {
            'avg_pause_duration': 0,
            'max_pause_duration': 0,
            'pause_frequency': 0,
            'std_pause_duration': 0,
            'long_pause_count': 0,
            'long_pause_ratio': 0
        }
    
