class AudioAnalyzer:
    """Audio deepfake detection analyzer"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioDeepfakeDetector().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def extract_features(self, audio_path, sr=16000, n_mels=128, duration=5):
        """Extract mel-spectrogram features from audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Extract additional features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Combine features
            features = np.concatenate([
                mel_spec_db,
                mfcc,
                spectral_contrast,
                chroma
            ], axis=0)
            
            return features
        except Exception as e:
            raise Exception(f"Error extracting audio features: {str(e)}")
    
    def analyze(self, audio_path):
        """Analyze audio file for deepfake detection"""
        try:
            # Extract features
            features = self.extract_features(audio_path)
            
            # Prepare for model
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output, attention = self.model(features_tensor)
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            is_deepfake = bool(prediction)
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': float(confidence),
                'probabilities': {
                    'real': float(probabilities[0][0]),
                    'fake': float(probabilities[0][1])
                },
                'analysis_type': 'audio_spectrogram_analysis'
            }
        except Exception as e:
            raise Exception(f"Error analyzing audio: {str(e)}")


class TextAnalyzer:
    """BERT-based AI-generated text detection analyzer"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TextAIDetector().to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def extract_statistical_features(self, text):
        """Extract statistical features from text"""
        words = text.split()
        sentences = text.split('.')
        
        features = [
            len(words),  # Word count
            len(sentences),  # Sentence count
            np.mean([len(w) for w in words]) if words else 0,  # Avg word length
            len(set(words)) / len(words) if words else 0,  # Lexical diversity
            text.count(',') / len(words) if words else 0,  # Comma density
            text.count('the') / len(words) if words else 0,  # Common word freq
            np.std([len(s) for s in sentences]) if len(sentences) > 1 else 0,  # Sentence length variation
            sum(1 for c in text if c.isupper()) / len(text) if text else 0,  # Capitalization ratio
            sum(1 for c in text if c in '!?') / len(text) if text else 0,  # Punctuation ratio
            len(text)  # Total character count
        ]
        
        return np.array(features, dtype=np.float32)
    
    def analyze(self, text, max_length=512):
        """Analyze text for AI generation detection"""
        try:
            # Tokenize
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Statistical features
            stat_features = self.extract_statistical_features(text)
            stat_tensor = torch.FloatTensor(stat_features).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_ids, attention_mask, stat_tensor)
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            is_ai_generated = bool(prediction)
            
            return {
                'is_ai_generated': is_ai_generated,
                'confidence': float(confidence),
                'probabilities': {
                    'human': float(probabilities[0][0]),
                    'ai': float(probabilities[0][1])
                },
                'text_length': len(text),
                'word_count': len(text.split()),
                'analysis_type': 'bert_linguistic_analysis'
            }
        except Exception as e:
            raise Exception(f"Error analyzing text: {str(e)}")